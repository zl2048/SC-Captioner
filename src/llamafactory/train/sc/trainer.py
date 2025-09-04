# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F
from transformers import Trainer, GenerationConfig
from trl.trainer.online_dpo_trainer import OnlineDPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


from typing import Any
import torch.nn as nn

from .reward_utils import *
from capture_metric.stop_words import stop_words_list
from sentence_transformers import SentenceTransformer
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import contextlib
import io
from nltk.tokenize import sent_tokenize
from .capture import CAPTURE
from transformers import logging as hf_logging
from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases
import collections
import difflib
import numpy as np


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomSCTrainer(OnlineDPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        #reward_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_type: Optional[str] = None,
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False
        self.model_type = model_type

        self.ref_model = ref_model
        #self.reward_model = reward_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        # initialize reward metric
        self.synonym_matching = True
        self.stop_words_list = set(stop_words_list)
        text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # dpo hyperparams
        self._beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.stats = {
            "kl_div": [],
            "reward": [],
            "loss": [],
        }
        self.correction_instruction = "The previous response is not very good. Please review the objects, attributes and relations in the caption. Remove that not appear in the image and add missing ones in the previous caption. Directly output the final caption: "
        # initialize kl loss
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.9,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            use_cache=False,
        )

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()
        
        '''
        if reward_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model, "is_loaded_in_8bit", False) or getattr(reward_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)
                self.reward_model.eval()
        '''
                
        # 
        text_encoder = self.accelerator.prepare_model(text_encoder, evaluation_mode=True)
        text_encoder.eval()
        self.parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=text_encoder.device)
        self.capture = CAPTURE()
        self.capture.text_encoder = text_encoder
        
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _prepare_second_attempt_prompt(self, prompts, first_attempt):
        '''
        prompts: dict(input_ids, attention_mask, pixel_values, image_grid_thw)
        first_attempt: tensor(batch_size, seq_len)
        '''
        if self.model_type != 'llava':
            completed_correction_instruction = f"\n<|im_start|>user\n{self.correction_instruction}<|im_end|>\n<|im_start|>assistant\n"
        else:
            completed_correction_instruction = f" USER: {self.correction_instruction} ASSISTANT:"
        correction_instruction = (
            self.tokenizer.encode(completed_correction_instruction, return_tensors="pt", add_special_tokens=False)
            .repeat(prompts["input_ids"].shape[0], 1)
            .to(first_attempt.device)
        )

        second_attempt_input_ids = torch.cat([first_attempt, correction_instruction], dim=1)

        second_attempt_attention_mask = torch.ones_like(second_attempt_input_ids)
        second_attempt_attention_mask[second_attempt_input_ids == self.tokenizer.pad_token_id] = 0
        

        return {"input_ids": second_attempt_input_ids, "attention_mask": second_attempt_attention_mask}
    
    @override
    def _generate_completions(self, model, prompts):
        unwrapped_model = self.accelerator.unwrap_model(model)
        # Generate first attempt
        
        if self.model_type != "llava":
            first_attempt = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                pixel_values=prompts["pixel_values"],
                image_grid_thw = prompts["image_grid_thw"],
                generation_config=self.generation_config,
            )
        else:
            first_attempt = unwrapped_model.generate(
                input_ids=prompts["input_ids"],
                attention_mask=prompts["attention_mask"],
                pixel_values=prompts["pixel_values"],
                generation_config=self.generation_config,
            )
        
        # Prepare input for second attempt
        second_attempt_prompt = self._prepare_second_attempt_prompt(prompts, first_attempt)

        # Generate second attempt
        if self.model_type != "llava":

            second_attempt = unwrapped_model.generate(
                input_ids=second_attempt_prompt["input_ids"],
                attention_mask=second_attempt_prompt["attention_mask"],
                pixel_values=prompts["pixel_values"],
                image_grid_thw = prompts["image_grid_thw"],
                generation_config=self.generation_config,
            )
        else:
            second_attempt = unwrapped_model.generate(
                input_ids=second_attempt_prompt["input_ids"],
                attention_mask=second_attempt_prompt["attention_mask"],
                pixel_values=prompts["pixel_values"],
                generation_config=self.generation_config,
            )

        second_attempt_gt = torch.cat((prompts['input_ids'], prompts["second_completion_input_ids"]), dim=1)

        return first_attempt, second_attempt, second_attempt_gt
    
    @override
    def _process_completion(self, completion, prompts):
        # generate a mask according to completion
        completion_mask = torch.ones_like(completion)
        completion_mask[completion == self.tokenizer.pad_token_id] = 0
        return {
            "input_ids": completion,
            "attention_mask": completion_mask,
        }
        
    
    def compute_rewards(self, first_attempt_texts, second_attempt_texts, prompts):
        return_reward = torch.zeros(len(second_attempt_texts)).to(prompts["input_ids"].device)
        sum_reward = 0
        for i in range(len(second_attempt_texts)):
            '''
            first process second-turn caption
            '''
            
            text_ref = second_attempt_texts[i]
            sentences_ref = sent_tokenize(text_ref)
            with torch.no_grad():
                with io.StringIO() as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        graph_obj = self.parser.parse(sentences_ref, beam_size=5, return_text=False,max_output_len=128, max_input_len=512)
            # form into set
            objects, attributes, relations, relations_original_ref, attributes_original_ref, objects_original_ref, attributes_ = merge_sentence_results(graph_obj, self.capture.text_processor)
            objects_ref = [object for object in objects if object not in self.stop_words_list and self.capture.isinsentence(object, text_ref)]
            attributes_ref = {k: v for k,v in attributes.items() if self.capture.isinsentence(k,text_ref)}    
            relations_ref = set([relation for relation in relations if self.capture.isinsentence(relation[0], text_ref) and self.capture.isinsentence(relation[2], text_ref) ])
            '''
            process first-turn caption
            '''
            text_rejected = first_attempt_texts[i]
            sentences_rejected = sent_tokenize(text_rejected)
            with torch.no_grad():
                with io.StringIO() as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        graph_obj = self.parser.parse(sentences_rejected, beam_size=5, return_text=False,max_output_len=128, max_input_len=512)
            # form into set
            objects, attributes, relations, relations_original_rejected, attributes_original_rejected, objects_original_rejected, attributes_ = merge_sentence_results(graph_obj, self.capture.text_processor)
            objects_rejected = [object for object in objects if object not in self.stop_words_list and self.capture.isinsentence(object, text_rejected)]
            attributes_rejected = {k: v for k,v in attributes.items() if self.capture.isinsentence(k,text_rejected)}    
            relations_rejected = set([relation for relation in relations if self.capture.isinsentence(relation[0], text_rejected) and self.capture.isinsentence(relation[2], text_rejected) ])

            '''
            process gt caption
            '''
            text_gt = prompts["chosen_text"][i]
            sentences_gt = sent_tokenize(text_gt)
            with torch.no_grad():
                with io.StringIO() as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        graph_obj = self.parser.parse(sentences_gt, beam_size=5, return_text=False,max_output_len=128, max_input_len=512)
            # form into set
            objects, attributes, relations, relations_original_gt, attributes_original_gt, objects_original_gt, attributes_ = merge_sentence_results(graph_obj, self.capture.text_processor)
            objects_gt = [object for object in objects if object not in self.stop_words_list and self.capture.isinsentence(object, text_gt)]
            attributes_gt = {k: v for k,v in attributes.items() if self.capture.isinsentence(k,text_gt)}    # k in text_all_nouns and k not in self.stop_words_list}
            relations_gt = set([relation for relation in relations if self.capture.isinsentence(relation[0], text_gt) and self.capture.isinsentence(relation[2], text_gt) ])

            '''
            compute reward
            '''

            # calculate the additions and removals made by models between first and second turns.
            removed_objects_ref, added_objects_ref, removed_relations_ref, added_relations_ref, removed_attributes_ref, added_attributes_ref = get_revision(
                objects_original_rejected,
                objects_original_ref,
                attributes_original_rejected,
                attributes_original_ref,
                relations_original_rejected,
                relations_original_ref,
                text_rejected,
                text_ref,
                self.capture.text_encoder,
                stop_words=True
            )
            # initialize rewards: hard and soft
            bonus = 0 
            rewards_soft = 0

            # form as list
            removed_objects_ref_list = list(removed_objects_ref)
            added_objects_ref_list = list(added_objects_ref)
            objects_gt_list = list(objects_original_gt)

            # removed objects & gt (if not similar, it is good removal, if similar, it is bad removal)
            if removed_objects_ref_list and objects_gt_list:
                with io.StringIO() as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        removed_objects_ref_features, gt_objects_features = encode_phrases(self.capture.text_encoder, removed_objects_ref_list, objects_gt_list, batch_size=4)
                sim_mat_1 = removed_objects_ref_features.dot(gt_objects_features.T)
                max_sim_1 = sim_mat_1.max(axis=1)
                rewards_soft += (0.6-max_sim_1).mean()
                bonus += min(np.sum(max_sim_1<0.6)*0.25,0.75)
                bonus -= min(np.sum(max_sim_1>0.7)*0.25,0.75)
                
            
            # added objects & gt (if similar, it is good addition, if not similar, it is bad addition)
            if added_objects_ref_list and objects_gt_list:
                with io.StringIO() as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        added_objects_ref_features, gt_objects_features = encode_phrases(self.capture.text_encoder, added_objects_ref_list, objects_gt_list, batch_size=4)
                sim_mat_2 = added_objects_ref_features.dot(gt_objects_features.T)
                max_sim_2 = sim_mat_2.max(axis=1)
                rewards_soft += (max_sim_2-0.55).mean()
                bonus -= min(np.sum(max_sim_2<0.6)*0.25,0.75) 
                bonus += min(np.sum(max_sim_2>0.7)*0.25,0.75)
            
            # removed attributes & gt (if not similar, it is good removal, if similar, it is bad removal)
            for key in removed_attributes_ref:
                if key in attributes_gt:
                    attributes_removed_list = list(removed_attributes_ref[key])
                    gt_attributes_list = list(attributes_gt[key])
                    with io.StringIO() as f:
                        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            attributes_removed_features, gt_attributes_features = encode_phrases(self.capture.text_encoder, attributes_removed_list, gt_attributes_list, batch_size=4)
                    sim_mat_3 = attributes_removed_features.dot(gt_attributes_features.T)
                    max_sim_3 = sim_mat_3.max(axis=1)
                    bonus -= min(np.sum(max_sim_3>0.7)*0.15, 0.45)
            
            # added attributes & gt (if similar, it is good addition, if not similar, it is bad addition)
            for key in added_attributes_ref:
                if key in attributes_gt:
                    attributes_added_list = list(added_attributes_ref[key])
                    gt_attributes_list = list(attributes_gt[key])
                    with io.StringIO() as f:
                        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                            attributes_added_features, gt_attributes_features = encode_phrases(self.capture.text_encoder, attributes_added_list, gt_attributes_list, batch_size=4)
                    sim_mat_4 = attributes_added_features.dot(gt_attributes_features.T)
                    max_sim_4 = sim_mat_4.max(axis=1)
                    bonus += min(np.sum(max_sim_4>0.7)*0.15, 0.45)

            if text_rejected == text_ref:
                bonus = -1

            if len(text_rejected)*2 < len(text_ref):
                bonux = -3

            return_reward[i] = F.sigmoid(torch.tensor(rewards_soft*2 + bonus))-0.5
            a=1    
            
            
        return return_reward
    def _compute_stage2_loss(self, model, ref_model, first_attempt, second_attempt, second_attempt_gt, prompts, ground_truth_completions):
        context_length = prompts["input_ids"].shape[1]
        length_before2 = prompts["length_before2"]

        # Compute logprobs for first attempt
        if self.model_type != "llava":
            first_attempt_logits = model(
                first_attempt["input_ids"], 
                attention_mask=first_attempt["attention_mask"],
                pixel_values=prompts["pixel_values"],
                image_grid_thw = prompts["image_grid_thw"],
            ).logits
        else:
            first_attempt_logits = model(
                first_attempt["input_ids"], 
                attention_mask=first_attempt["attention_mask"],
                pixel_values=prompts["pixel_values"],
            ).logits
        first_attempt_logprobs = F.log_softmax(first_attempt_logits[:, context_length - 1 : -1], dim=-1)

        # Compute logprobs for second attempt
        if self.model_type != "llava":
            second_attempt_logits = model(
                second_attempt["input_ids"], 
                attention_mask=second_attempt["attention_mask"],
                pixel_values=prompts["pixel_values"],
                image_grid_thw = prompts["image_grid_thw"],
            ).logits
        else:
            second_attempt_logits = model(
                second_attempt["input_ids"], 
                attention_mask=second_attempt["attention_mask"],
                pixel_values=prompts["pixel_values"],
            ).logits
        second_attempt_logprobs = F.log_softmax(second_attempt_logits[:, context_length - 1 : -1], dim=-1)  # (bs,seq_len,tokenizer_size)
        
        # Compute KL divergence for first 
        with torch.no_grad():
            if self.model_type != "llava":
                ref_first_attempt_logits = ref_model(
                    first_attempt["input_ids"], 
                    attention_mask=first_attempt["attention_mask"],
                    pixel_values=prompts["pixel_values"],
                    image_grid_thw = prompts["image_grid_thw"],
                ).logits
            else:
                ref_first_attempt_logits = ref_model(
                    first_attempt["input_ids"], 
                    attention_mask=first_attempt["attention_mask"],
                    pixel_values=prompts["pixel_values"],
                ).logits
            ref_first_attempt_logprobs = F.log_softmax(ref_first_attempt_logits[:, context_length - 1 : -1], dim=-1)
        

        # Create a mask for non-padding tokens
        non_padding_mask = (first_attempt["input_ids"][:, context_length:] != self.tokenizer.pad_token_id).float()
        second_attempt_non_padding_mask = (second_attempt["input_ids"][:, context_length:] != self.tokenizer.pad_token_id).float()

        # Gather the log probabilities of the actual tokens
        first_attempt_tokens = first_attempt["input_ids"][:, context_length:]
        first_attempt_logprobs = torch.gather(first_attempt_logprobs, 2, first_attempt_tokens.unsqueeze(-1)).squeeze(
            -1
        )  # (bs,seq_len)
        ref_first_attempt_logprobs = torch.gather(
            ref_first_attempt_logprobs, 2, first_attempt_tokens.unsqueeze(-1)
        ).squeeze(-1)

        second_attempt_tokens = second_attempt["input_ids"][:, context_length:]

        second_attempt_logprobs = torch.gather(
            second_attempt_logprobs, 2, second_attempt_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        first_attempt_logprobs = torch.masked_fill(first_attempt_logprobs, ~non_padding_mask.bool(), 0)
        ref_first_attempt_logprobs = torch.masked_fill(ref_first_attempt_logprobs, ~non_padding_mask.bool(), 0)
        second_attempt_logprobs = torch.masked_fill(
            second_attempt_logprobs, ~second_attempt_non_padding_mask.bool(), 0
        )
            
        # begin compute loss
        beta_sc = 0.05
        # Compute KL divergence
        kl_div = (first_attempt_logprobs - ref_first_attempt_logprobs) * non_padding_mask
        kl_div = kl_div.sum(-1).mean()

        first_attempt_texts = prompts["first_response"]
        second_attempt_texts = prompts["second_response"]

        # prevent extra outputs, then calculate reward scores
        old_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        reward = self.compute_rewards(first_attempt_texts, second_attempt_texts, prompts)
        hf_logging.set_verbosity(old_level)
        
        # Compute REINFORCE loss with KL penalty
        policy_loss = -(second_attempt_logprobs.sum(-1) * reward).mean()
        kl_loss = beta_sc * kl_div
        loss = policy_loss + kl_loss


        self.stats["kl_div"].append(kl_loss.item())
        self.stats["reward"].append(reward.mean().item())
        self.stats["loss"].append(policy_loss.item())

        return loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        # Apply chat template and tokenize the input.
        # We do this on-the-fly to enable the use of reward models and policies with different tokenizers / chat templates.

        prompts = {
            "input_ids": inputs["prompt_input_ids"],
            "attention_mask": inputs["prompt_attention_mask"],
            "first_completion_input_ids": inputs["first_completion_input_ids"],
            "first_completion_attention_mask": inputs["first_completion_attention_mask"],
            "second_completion_input_ids": inputs["completion_input_ids"],
            "pixel_values" : inputs["pixel_values"],
            "chosen_text": inputs["chosen_text"],
            "rejected_text": inputs["rejected_text"],
        }
        if self.model_type != "llava":
            prompts["image_grid_thw"] = inputs["image_grid_thw"]
        ground_truth_completions = {
            "input_ids": inputs["completion_input_ids"],
            "attention_mask": inputs["completion_attention_mask"],
        }

        # Generate completions (both first and second attempts)
        first_attempt, second_attempt, second_attempt_gt = self._generate_completions(model, prompts)

        second_attempt_prompt = self._prepare_second_attempt_prompt(prompts, first_attempt)
        length_before2 = second_attempt_prompt["input_ids"].shape[1]
        prompts["second_response"] = self.tokenizer.batch_decode(second_attempt[:,length_before2:], skip_special_tokens=True)
        prompts["length_before2"] = length_before2

        context_length = prompts["input_ids"].shape[1]
        prompts["first_response"] = self.tokenizer.batch_decode(first_attempt[:,context_length:], skip_special_tokens=True)
        
        # Process completions
        first_attempt_data = self._process_completion(first_attempt, prompts)
        second_attempt_data = self._process_completion(second_attempt, prompts)
        second_attempt_data_gt = self._process_completion(second_attempt_gt, prompts)

        # Compute loss
        loss = self._compute_stage2_loss(
            model, self.ref_model, first_attempt_data, second_attempt_data, second_attempt_data_gt, prompts, ground_truth_completions
        )

        if self.args.n_gpu > 1:
            loss = loss.mean()  

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

    
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        # loss, generated_tokens, _ = super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

    
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels
    
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
