"""
    Copyright (2024) CAPTURE project Authors 

    Licensed under the Apache License, Version 2.0 (the "License"); 
    you may not use this file except in compliance with the License. 
    You may obtain a copy of the License at 

        http://www.apache.org/licenses/LICENSE-2.0 

    Unless required by applicable law or agreed to in writing, software 
    distributed under the License is distributed on an "AS IS" BASIS, 
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
    See the License for the specific language governing permissions and 
    limitations under the License.
"""


import functools
import tabulate
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import collections
import torch
import tqdm
import contextlib
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import multiprocessing
from statistics import mean

from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases


_tabulate_format = tabulate.TableFormat(
    lineabove=tabulate.Line("+", "-", "+", "+"),
    linebelowheader=tabulate.Line("|", "-", "+", "|"),
    linebetweenrows=None,
    linebelow=tabulate.Line("+", "-", "+", "+"),
    headerrow=tabulate.DataRow("|", "|", "|"),
    datarow=tabulate.DataRow("|", "|", "|"),
    padding=1, with_header_hide=None
)

def tprint(graph, file=None):
    """
    Print a scene graph as a table.
    The printed strings contain essential information about the parsed scene graph.
    """
    assert isinstance(graph, dict), 'Input must be a dictionary'
    _print = functools.partial(print, file=file)

    _print('Entities:')
    entities_data = [
        [e['head'].lower(), e.get('quantity', ''), ','.join(e.get('attributes', set()))]
        for e in graph['entities']
    ]
    _print(tabulate.tabulate(entities_data, headers=['Entity', 'Quantity', 'Attributes'], tablefmt=_tabulate_format))

    _print('Relations:')
    relations_data = [
        [
            graph['entities'][rel['subject']]['head'].lower(),
            rel['relation'].lower(),
            graph['entities'][rel['object']]['head'].lower()
        ]
        for rel in graph['relations']
    ]
    _print(tabulate.tabulate(relations_data, headers=['Subject', 'Relation', 'Object'], tablefmt=_tabulate_format))


def merge_sentence_results(results, text_processor):
    # from IPython import embed; embed()
    objects, attributes, relations = set(), collections.defaultdict(set), set()
    for result in results:
        for entity in result['entities']:
            lemmatized_obj = text_processor.normalize_word(entity['head'], wordnet.NOUN)
            objects.add(lemmatized_obj)
            for attribute in entity['attributes']:
                attribute = text_processor.normalize_word(attribute, wordnet.ADJ)
                if ' of' in attribute:
                    continue
                attributes[lemmatized_obj].add(attribute)
        for relation in result['relations']:
            relations.add((
                text_processor.normalize_word(result['entities'][relation['subject']]['head'], wordnet.NOUN), 
                relation['relation'], 
                text_processor.normalize_word(result['entities'][relation['object']]['head'], wordnet.NOUN)
            ))

    return objects, attributes, relations


def are_tuples_match(synsets1, synsets2):
    """
    Determine if two lists of synsets have non-empty intersections for corresponding elements.

    :param synsets1: First list of synsets.
    :param synsets2: Second list of synsets.
    :return: True if all corresponding synsets have a non-empty intersection, False otherwise.
    """

    return len(synsets1) == len(synsets2) and all(s1.intersection(s2) for s1, s2 in zip(synsets1, synsets2))


def get_synonyms(word):
    synsets = wordnet.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


def set_mp_context(expected_context='spawn'):
    default_context_name = torch.multiprocessing.get_context().get_start_method()
    if default_context_name != expected_context:
        torch.multiprocessing.set_start_method('spawn', force=True)
    return


class TextProcessor:
    def __init__(self) -> None:
        self.wnl = WordNetLemmatizer()

    def normalize_word(self, word, pos):
        return self.wnl.lemmatize(word, pos=pos)


class CAPTURE:
    def __init__(
        self, 
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.2,
        synonym_matching: bool = True,
        soft_matching: bool = True,
        stop_words: bool = True,
        eps: float = 1e-6,
    ):
        """
        Args:
            alpha (`float`, *optional*, defaults to be 0.5):
                The ratio of object F1 score considered in CAPTURE score computation.
            beta (`float`, *optional*, defaults to be 0.5):
                The ratio of attribute F1 score considered in CAPTURE score computation.
                The summation of alpha and beta must equals to 1.
            gamma (`float`, *optional*, defaults to be 0.2):
                The ratio of relation F1 score considered in CAPTURE score computation.
            synonym_matching (`bool`, *optional*, defaults to be True):
                Controls whether to use synonym_matching for visual elements mathcing. 
            soft_matching (`bool`, *optional*, defaults to be True):
                Controls whether to use soft_matching for visual elements mathcing.   
            stop_words (`bool`, *optional*, defaults to be True):
                Controls whether to use stop words object elements filtering.  
            eps (`float`, *optional*, defaults to be 1e-6):
                A small number to avoid division by zero when computing precision, recall and F1. 
        """
        self.alpha = alpha
        self.beta = beta
        assert self.alpha + self.beta == 1.
        self.gamma = gamma
        self.parser = None
        self.text_processor=TextProcessor()
        self.synonym_matching = synonym_matching

        if stop_words:
            from capture_metric.stop_words import stop_words_list
            self.stop_words_list = set(stop_words_list)
        else:
            self.stop_words_list = set([])

        self.eps = eps

        self.soft_matching = soft_matching
        if self.soft_matching:
            self.text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to('cuda:0').eval()
        
    
    def compute_synonyms_score(self, word1, word2):
        # in case word1 or word2 consists of multiple words
        if word1 in word2 or word2 in word1:
            return 1
        elif len(word1.split()) > 0 or len(word2.split() > 0):
            word1 = '_'.join(word1.split())
            word2 = '_'.join(word2.split())

        synonyms1 = get_synonyms(word1)
        synonyms2 = get_synonyms(word2)
        iou = len(synonyms1.intersection(synonyms2)) / (len(synonyms1.union(synonyms2)) + self.eps)
        return iou


    def compute_match(self, all_cand, all_gt):
        total_match = 0
        matched_cand_indices, matched_ref_indices = set(), set()
        for ii, cand in enumerate(all_cand):
            for jj, ref in enumerate(all_gt):
                if cand == ref and jj not in matched_ref_indices:
                    matched_cand_indices.add(ii)
                    matched_ref_indices.add(jj)
                    # print(cand, ref)
                    total_match += 1
                    break

        if self.synonym_matching:
            for ii, cand in enumerate(all_cand):
                if ii not in matched_cand_indices:
                    for jj, ref in enumerate(all_gt):
                        if jj not in matched_ref_indices and self.compute_synonyms_score(cand, ref) > 0.:
                            matched_cand_indices.add(ii)
                            matched_ref_indices.add(jj)
                            # print(cand, ref)
                            total_match += 1
                            break
        
        remained_cands = [cand for i, cand in enumerate(all_cand) if i not in matched_cand_indices]
        remained_refs = [gt for j, gt in enumerate(all_gt) if j not in matched_ref_indices]
        cand_match = total_match
        ref_match = total_match
        if self.soft_matching and len(remained_cands) > 0 and len(remained_refs) > 0:
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    remained_cands_features, remained_refs_features = encode_phrases(self.text_encoder, remained_cands, remained_refs, batch_size=4)
            sim_mat = remained_cands_features.dot(remained_refs_features.T)
            remained_cands_match = np.sum(np.max(sim_mat, axis=1))
            remained_refs_match = np.sum(np.max(sim_mat, axis=0))
            cand_match = total_match + remained_cands_match
            ref_match = total_match + remained_refs_match

        return total_match, cand_match, ref_match


    def get_all_lemmatized_nouns(self, text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [self.text_processor.normalize_word(token, pos=wordnet.NOUN) for token, tag in tagged if tag.startswith('NN')]
        return nouns


    def compute_objects(self, gt_parsed, cand_parsed, extra_objects):
        gt_objects, gt_attributes, gt_relations = gt_parsed
        cand_objects, cand_attributes, cand_relations = cand_parsed
        gt_objects_added = list(set(gt_objects+extra_objects))

        # Objects
        _, _, object_ref_match = self.compute_match(cand_objects, gt_objects)
        _, object_cand_match, _ = self.compute_match(cand_objects, gt_objects_added)
        object_precision, object_recall = object_cand_match / (len(cand_objects) + self.eps), object_ref_match / (len(gt_objects) + self.eps)
        object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall + self.eps)


        return object_precision, object_recall, object_f1, \
    
    def compute_match_find(self, all_cand, all_gt):
        
        with io.StringIO() as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                cands_features, gts_features = encode_phrases(self.text_encoder, all_cand, all_gt, batch_size=4)
        sim_mat = cands_features.dot(gts_features.T)
        match_score = np.max(sim_mat, axis=0)
        match_place = np.argmax(sim_mat, axis=0)
        
        a=1
        

        return match_score, match_place
    
    def compute_attributes(self, gt_parsed, cand_parsed, extra_attributes):
        _, gt_attributes, gt_relations = gt_parsed
        _, cand_attributes, cand_relations = cand_parsed
        for key in extra_attributes:
            extra_attributes[key] = set(extra_attributes[key])

        gt_attributes_list = [[key, value] for key, value in gt_attributes.items()]
        cand_attributes_list = [[key, value] for key, value in cand_attributes.items()]
        extra_attributes_list = [[key, value] for key, value in extra_attributes.items()]
        
        gt_objects_att = [key for key, value in gt_attributes_list]
        cand_objects_att = [key for key, value in cand_attributes_list]
        extra_objects_att = [key for key, value in extra_attributes_list]

        match_score, match_place = self.compute_match_find(cand_objects_att, gt_objects_att)
        
        number_attributes = 0
        total_match_score = 0
        for i in range(len(gt_objects_att)):
            this_cand_attributes = list(cand_attributes_list[match_place[i]][1])
            this_gt_attributes = list(gt_attributes_list[i][1])
            object_similarity = match_score[i]
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    cands_features, gts_features = encode_phrases(self.text_encoder, this_cand_attributes, this_gt_attributes, batch_size=4)
            sim_mat = cands_features.dot(gts_features.T)
            this_gt_attributes_score = np.max(sim_mat, axis=0)
            number_attributes += object_similarity*this_gt_attributes_score.size
            total_match_score += object_similarity*np.sum(this_gt_attributes_score)
            
            a=1

        attribute_recall = total_match_score / (number_attributes + self.eps)

        match_score_extra, match_place_extra = self.compute_match_find(extra_objects_att, cand_objects_att)  

        number_attributes_cand = 0
        total_match_score_precision = 0
        for i in range(len(cand_objects_att)):
            this_extra_attributes = list(extra_attributes_list[match_place_extra[i]][1])
            this_cand_attributes = list(cand_attributes_list[i][1])
            object_similarity = match_score_extra[i]
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    extras_features, cands_features = encode_phrases(self.text_encoder, this_extra_attributes, this_cand_attributes, batch_size=4)
            sim_mat = extras_features.dot(cands_features.T)
            this_gt_attributes_score = np.max(sim_mat, axis=0)
            number_attributes_cand += object_similarity*this_gt_attributes_score.size
            total_match_score_precision += object_similarity*np.sum(this_gt_attributes_score)
            
            a=1

        attribute_precision = total_match_score_precision / (number_attributes_cand + self.eps)
        attribute_f1 = 2 * attribute_precision * attribute_recall / (attribute_precision + attribute_recall + self.eps)



        return attribute_precision, attribute_recall, attribute_f1
                
                

    def sample_to_parse_results(self, sample):
        sample_index, text = sample[0], sample[1]
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            print(e)
            print(f"text: {text}")
            import pdb; pdb.set_trace()
        with torch.no_grad():
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    graph_obj = self.parser.parse(sentences, beam_size=5, return_text=False,max_output_len=128)
        
        objects, attributes, relations = merge_sentence_results(graph_obj, self.text_processor)
        text_all_nouns = set(self.get_all_lemmatized_nouns(text))
        objects = [object for object in objects if object not in self.stop_words_list and (object in text_all_nouns or all([piece in text_all_nouns for piece in object.split(' ')]))]
        attributes = {k: v for k,v in attributes.items() if (k in text_all_nouns or all([piece in text_all_nouns for piece in k.split(' ')]))}    # k in text_all_nouns and k not in self.stop_words_list}
        relations = set([relation for relation in relations if (relation[0] in text_all_nouns or all([piece in text_all_nouns for piece in relation[0].split(' ')])) and (relation[2] in text_all_nouns or all([piece in text_all_nouns for piece in relation[2].split(' ')])) ])  
        return sample_index, objects, attributes, relations


    def parse_samples(self, samples, device, desc=""):
        torch.cuda.set_device(int(str(device)[-1]))
        if self.parser is not None and hasattr(self.parser, 'device') and self.parser.device == device:
            pass
        else:
            if self.parser is not None:
                print(f"self.parser.device {self.parser.device} device {device}")
            if torch.cuda.is_available():
                self.parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)
            else:
                self.parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device='cpu')
            self.parser.model.eval()
        parsed_samples = []
        for sample in tqdm.tqdm(samples, desc=desc + ' ' + str(device)):
            parsed_sample = self.sample_to_parse_results(sample)
            parsed_samples.append(parsed_sample)
        return parsed_samples


    def process_samples_multiprocessing(self, partitioned_data, desc="parsing"):
        set_mp_context()
        with multiprocessing.Pool(processes=torch.cuda.device_count()) as pool:
            futures = []
            for idx, this_partitioned_data in enumerate(partitioned_data):
                future = pool.apply_async(self.parse_samples, args=(this_partitioned_data, torch.device(f'cuda:{idx}'), desc))
                futures.append(future)
            all_parsed = []
            for future in futures:
                results = future.get()
                all_parsed.extend(results)
        # all_parsed.sort(key=lambda x: x[0])
        # all_parsed = [(res[1], res[2], res[3]) for res in all_parsed]
        # return all_parsed

        all_parsed_dict = collections.defaultdict(list)
        for parsed_sample in all_parsed:
            all_parsed_dict[parsed_sample[0]].append(parsed_sample[1:])
        return all_parsed_dict
        

    def compute_score(self, gts, res, prev_gt_parsed=None, prev_cand_parsed=None, return_parse_results=False, extra_objects=None, extra_attributes=None):
        gts = [(sample_key, gt) for sample_key, sample_gts in gts.items() for gt in sample_gts]
        cands = [(sample_key, sample_res[0]) for sample_key, sample_res in res.items()]

        def partition_data(data):
            num_chunk = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
            chunk_size = len(data) // num_chunk
            partitioned_data = []
            start = 0
            for i in range(num_chunk):
                end = start + chunk_size
                if i < len(data) % num_chunk:
                    end += 1
                partitioned_data.append(data[start:end])
                start = end
            return partitioned_data

        if prev_cand_parsed is None:
            partitioned_data = partition_data(cands)
            cand_parsed = self.process_samples_multiprocessing(partitioned_data, desc='parsing cand')
        else:
            print("parsing cand skip")
            cand_parsed = prev_cand_parsed

        if prev_gt_parsed is None:
            partitioned_data = partition_data(gts)
            gt_parsed = self.process_samples_multiprocessing(partitioned_data, desc='parsing gt')
        else:
            print("parsing gt skip")
            gt_parsed = prev_gt_parsed

        scores = []
        parse_results = []
        object_precisions = []
        object_recalls = []
        object_f1s = []
        attribute_precisions = []
        attribute_recalls = []
        attribute_f1s = []
        for sample_key in tqdm.tqdm(gt_parsed.keys(), desc="computing score"):
            sample_gt_parsed, sample_cand_parsed = gt_parsed[sample_key], cand_parsed[sample_key][0]
            sample_extra_objects = extra_objects[sample_key]
            sample_extra_attributes = extra_attributes[sample_key]
            if len(sample_gt_parsed[0][1]) == 0 or len(sample_cand_parsed[1]) == 0: continue
            object_results = self.compute_objects(sample_gt_parsed[0], sample_cand_parsed, sample_extra_objects)
            
            
            attributes_results = self.compute_attributes(sample_gt_parsed[0], sample_cand_parsed, sample_extra_attributes)
            
            object_precisions.append(object_results[0])
            object_recalls.append(object_results[1])
            object_f1s.append(object_results[2])
            attribute_precisions.append(attributes_results[0])
            attribute_recalls.append(attributes_results[1])
            attribute_f1s.append(attributes_results[2])
            

        object_precision = sum(object_precisions) / len(object_precisions)
        object_recall = sum(object_recalls) / len(object_recalls)
        object_f1 = sum(object_f1s) / len(object_f1s)
        attribute_precision = sum(attribute_precisions) / len(attribute_precisions)
        attribute_recall = sum(attribute_recalls) / len(attribute_recalls)
        attribute_f1 = sum(attribute_f1s) / len(attribute_f1s)

        if return_parse_results:
            return object_precision, object_recall, object_f1, attribute_precision, attribute_recall, attribute_f1, cand_parsed
        else:
            return object_precision, object_recall, object_f1, attribute_precision, attribute_recall, attribute_f1
    




if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")

    refs = {
        'example_0': [
            "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. ",
            "The image depicts a busy city street with cars running in the foreground, including a red car and a white truck. The street is surrounded by green trees. In the backgound of the image, modern edifices and a clock tower stand under a clear blue sky. "
        ],
    }
    preds = {
        'example_0': [
            "The image shows a red car, a white truck and other automobiles running on a city road. Pedestrians are walking on the side. Tall buildings can be seen under a clear blue sky."
        ]
    }
    assert refs.keys() == preds.keys()

    evaluator = CAPTURE()
    score = evaluator.compute_score(refs, preds)
    print(f"CAPTURE score: {score}")







