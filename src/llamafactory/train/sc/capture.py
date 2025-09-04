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
from nltk.util import ngrams
import collections
import torch
import tqdm
import contextlib
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import multiprocessing
from statistics import mean
import re
from nltk import pos_tag, word_tokenize

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
    relations_original, attributes_original, objects_original = set(), set(), set()
    for result in results:
        for entity in result['entities']:
            objects_original.add(entity['head'])
            lemmatized_obj = text_processor.normalize_word(entity['head'], wordnet.NOUN)
            objects.add(lemmatized_obj)
            for attribute in entity['attributes']:
                attributes_original.add(attribute)
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
            relations_original.add((
                result['entities'][relation['subject']]['head'], 
                relation['relation'], 
                result['entities'][relation['object']]['head']
            ))

    return objects, attributes, relations, relations_original, attributes_original, objects_original


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

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ  # Adjective
        elif treebank_tag.startswith('V'):
            return wordnet.VERB  # Verb
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN  # Noun
        elif treebank_tag.startswith('R'):
            return wordnet.ADV  # Adverb
        else:
            return ''  

    def pos_tagging(self, sentence):
        tokenized_sentence = word_tokenize(sentence)
        tagged_sentence = pos_tag(tokenized_sentence)
        return tagged_sentence

    def lemmatize_sentence(self, sentence):
        lemmatizer = WordNetLemmatizer()
        tagged_sentence = self.pos_tagging(sentence)
        lemmatized_sentence = []
        for word, tag in tagged_sentence:
            pos = self.get_wordnet_pos(tag)
            if pos:
                lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
            else:
                lemmatized_word = word  
            lemmatized_sentence.append(lemmatized_word)
        return ' '.join(lemmatized_sentence)


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
        #if self.soft_matching:
            #self.text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to('cuda:0').eval()
            
        
    
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
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        nouns = [self.text_processor.normalize_word(token, pos=wordnet.NOUN) for token, tag in tagged if tag.startswith('NN')]
        return nouns


    def compute_f_score(self, gt_parsed, cand_parsed):
        gt_objects, gt_attributes, gt_relations = gt_parsed
        cand_objects, cand_attributes, cand_relations = cand_parsed

        # Objects
        object_match, object_cand_match, object_ref_match = self.compute_match(cand_objects, gt_objects)
        object_precision, object_recall = object_cand_match / (len(cand_objects) + self.eps), object_ref_match / (len(gt_objects) + self.eps)
        object_f1 = 2 * object_precision * object_recall / (object_precision + object_recall + self.eps)

        # Attributes
        gt_attributes_words, cand_attributes_words = [], []
        for k, v in gt_attributes.items():
            gt_attributes_words.extend(v)
        for k, v in cand_attributes.items():
            cand_attributes_words.extend(v)
        attribute_match, attribute_cand_match, attribute_ref_match = self.compute_match(cand_attributes_words, gt_attributes_words)
        attribute_precision, attribute_recall = attribute_cand_match / (len(cand_attributes_words) + self.eps), attribute_ref_match / (len(gt_attributes_words) + self.eps)
        attribute_f1 = 2 * attribute_precision * attribute_recall / (attribute_precision + attribute_recall + self.eps)

        gt_attributes_set, cand_attributes_set = set(), set()
        for key, values in gt_attributes.items():
            for value in values:
                gt_attributes_set.add((key, value))
        for key, values in cand_attributes.items():
            for value in values:
                cand_attributes_set.add((key, value))
 
        attribute_match = 0 
        matched_cand_indices, matched_ref_indices = set(), set() 
        for i, cand in enumerate(cand_attributes_set):
            for j, ref in enumerate(gt_attributes_set):
                if cand == ref and j not in matched_ref_indices:
                    matched_cand_indices.add(i)
                    matched_ref_indices.add(j)
                    attribute_match += 1
                    break

        if self.synonym_matching:
            for i, cand in enumerate(cand_attributes_set):
                if i not in matched_cand_indices:
                    for j, ref in enumerate(gt_attributes_set):
                        if j not in matched_ref_indices and all([self.compute_synonyms_score(cand_ele, ref_ele) > 0. for cand_ele, ref_ele in zip(cand, ref)]):
                            matched_cand_indices.add(i)
                            matched_ref_indices.add(j)
                            attribute_match += 1
                            break

        remained_cands = [' '.join(cand) for i, cand in enumerate(cand_attributes_set) if i not in matched_cand_indices]
        remained_refs = [' '.join(gt) for j, gt in enumerate(gt_attributes_set) if j not in matched_ref_indices]
        cands_match = attribute_match
        refs_match = attribute_match
        if self.soft_matching and len(remained_cands) > 0 and len(remained_refs) > 0:
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    remained_cands_features, remained_refs_features = encode_phrases(self.text_encoder, remained_cands, remained_refs, batch_size=4)
            sim_mat = remained_cands_features.dot(remained_refs_features.T)
            remained_cands_match = np.sum(np.max(sim_mat, axis=1))
            remained_refs_match = np.sum(np.max(sim_mat, axis=0))
            cands_match += remained_cands_match
            refs_match += remained_refs_match
        attribute_precision, attribute_recall = cands_match / (len(cand_attributes_set) + self.eps), refs_match / (len(gt_attributes_set) + self.eps)
        attribute_f1 = 2 * attribute_precision * attribute_recall / (attribute_precision + attribute_recall + self.eps)


        # Relations
        relation_match = 0
        matched_cand_indices, matched_ref_indices = set(), set()
        for i, cand in enumerate(cand_relations):
            for j, ref in enumerate(gt_relations):
                if cand == ref and j not in matched_ref_indices:
                    matched_cand_indices.add(i)
                    matched_ref_indices.add(j)
                    relation_match += 1
                    break
        
        if self.synonym_matching:
            for i, cand in enumerate(cand_relations):
                if i not in matched_cand_indices:
                    for j, ref in enumerate(gt_relations):
                        if j not in matched_ref_indices and all([self.compute_synonyms_score(cand_ele, ref_ele) > 0. for cand_ele, ref_ele in zip(cand, ref)]):
                            matched_cand_indices.add(i)
                            matched_ref_indices.add(j)
                            relation_match += 1
                            break
        
        remained_cands = [' '.join(cand) for i, cand in enumerate(cand_relations) if i not in matched_cand_indices]
        remained_refs = [' '.join(gt) for j, gt in enumerate(gt_relations) if j not in matched_ref_indices]
        cands_match = relation_match
        refs_match = relation_match
        if self.soft_matching and len(remained_cands) > 0 and len(remained_refs) > 0:
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    remained_cands_features, remained_refs_features = encode_phrases(self.text_encoder, remained_cands, remained_refs, batch_size=4)
            sim_mat = remained_cands_features.dot(remained_refs_features.T)
            remained_cands_match = np.sum(np.max(sim_mat, axis=1))
            remained_refs_match = np.sum(np.max(sim_mat, axis=0))
            cands_match += remained_cands_match
            refs_match += remained_refs_match

        relation_precision, relation_recall = cands_match / (len(cand_relations) + self.eps), refs_match / (len(gt_relations) + self.eps)
        relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall + self.eps)
        
        capture_score = self.alpha*object_f1 + self.beta*attribute_f1 + self.gamma * relation_f1
        capture_score /= (self.alpha + self.beta + self.gamma)
        # print(f"obj_f1: {object_f1}, attr_f1: {attribute_f1}, rel_f1: {relation_f1} capture: {capture_score}")
        avg_recall = (object_recall + attribute_recall + relation_recall) / 3

        return capture_score, object_precision, object_recall, object_f1, \
                attribute_precision, attribute_recall, attribute_f1, \
                relation_precision, relation_recall, relation_f1, avg_recall
                
    def isinsentence(self, object, sentence):
        length = len(object.split()) 
        
        words = nltk.word_tokenize(sentence)
        grams = list(ngrams(words, length))
        lemmatized_object = self.text_processor.lemmatize_sentence(object.lower())
        for gram in grams:
            phrase = ' '.join(gram)
            lemmatized_phrase = self.text_processor.lemmatize_sentence(phrase.lower())
            if lemmatized_phrase ==lemmatized_object or self.compute_synonyms_score(lemmatized_phrase, lemmatized_object) > 0:
                return True

        if length == 2:
            grams = list(ngrams(words, 1))
            lemmatized_object = self.text_processor.lemmatize_sentence(re.sub(r"\s+", "", object).lower())  # 合并
            for gram in grams:
                phrase = ' '.join(gram)
                lemmatized_phrase = self.text_processor.lemmatize_sentence(phrase.lower())
                if lemmatized_phrase ==lemmatized_object or self.compute_synonyms_score(lemmatized_phrase, lemmatized_object) > 0:
                    return True
        if length == 1:
            grams = list(ngrams(words, 2))
            lemmatized_object = self.text_processor.lemmatize_sentence(object.lower())
            for gram in grams:
                phrase = ''.join(gram)  
                lemmatized_phrase = self.text_processor.lemmatize_sentence(phrase.lower())
                if lemmatized_phrase ==lemmatized_object or self.compute_synonyms_score(lemmatized_phrase, lemmatized_object) > 0:
                    return True
            
        return False  


        
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
                    graph_obj = self.parser.parse(sentences, beam_size=5, return_text=False,max_output_len=128, max_input_len=512)
        
        objects, attributes, relations, relations_original, attributes_original, objects_original = merge_sentence_results(graph_obj, self.text_processor)
        
        '''
        for object in objects:
            if not self.isinsentence(object, text):
                print(f"Bad object: {object}, sentence: {text}")
        '''
        for object in objects:
            if object in self.stop_words_list:
                #print(f"Stop object: {object}, sentence: {text}")
                a=1
        #text_all_nouns = set(self.get_all_lemmatized_nouns(text))
        objects = [object for object in objects if object not in self.stop_words_list and self.isinsentence(object, text)]
        attributes = {k: v for k,v in attributes.items() if self.isinsentence(k,text)}    # k in text_all_nouns and k not in self.stop_words_list}
        relations = set([relation for relation in relations if self.isinsentence(relation[0], text) and self.isinsentence(relation[2], text) ])  
        return sample_index, objects, attributes, relations, relations_original, attributes_original, objects_original


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
    
    def remove_relations_from_sentence(self, sentence, relations, attributes, objects):
        # Step 1: Extract entities from the relations and sort them by length (longest first)
        sentence_original = sentence
        #sentence = self.text_processor.lemmatize_sentence(sentence)

        entities = set()
        for relation in relations:
            entities.add(relation[0])  # Add the first entity of each relation
            entities.add(relation[1])
            entities.add(relation[2])  # Add the second entity of each relation
        for attribute in attributes:
            entities.add(attribute)
        for object in objects:
            entities.add(object)
        
        # Sort entities by their length (longest first) to handle sub-phrases correctly
        sorted_entities = sorted(entities, key=lambda x: x.count(" "), reverse=True)

        result_dict = {}


        for entity in entities:
            word_count = len(entity.split())

            if word_count not in result_dict:
                result_dict[word_count] = []
            
            result_dict[word_count].append(entity)
        nltk.translate.bleu_score

        sorted_entity_dict = dict(sorted(result_dict.items(), key=lambda item: item[0], reverse=True))
        delete_list = []
        for key, entity_list in sorted_entity_dict.items():
            #print(key)
            words = nltk.word_tokenize(sentence_original)
            grams = list(ngrams(words, key))
            lemmatized_entity_list = [self.text_processor.lemmatize_sentence(entity.lower()) for entity in entity_list]
            for gram in grams:
                phrase = ' '.join(gram)
                lemmatized_phrase = self.text_processor.lemmatize_sentence(phrase.lower())
                if lemmatized_phrase in lemmatized_entity_list:
                    delete_list.append(phrase)

        
        # Step 2: Remove entities from the sentence
        for delete_phrase in delete_list:
            # Use a regular expression to find all occurrences of the entity and replace them
            sentence = re.sub(r'\b' + re.escape(delete_phrase) + r'\b', '', sentence, flags=re.IGNORECASE)
        
        # Step 3: Count remaining words
        remaining_words = sentence.strip().split()
        remaining_words_count = len(remaining_words)
        
        # Step 4: Count original words
        original_words = sentence_original.strip().split()
        original_words_count = len(original_words)
        
        # Step 5: Calculate the ratio
        if original_words_count == 0:
            return 0.0  # Avoid division by zero
        ratio = remaining_words_count / original_words_count
        
        return 1-ratio

    def compute_precision(self, gt_parsed, cand_parsed, extra_tags):
        gt_objects, gt_attributes, gt_relations = gt_parsed
        cand_objects, cand_attributes, cand_relations = cand_parsed
        gt_objects = set(gt_objects)
        cand_objects = set(cand_objects)
        extra_tags = set(extra_tags)
        # Objects
        all_cand, all_gt = cand_objects, gt_objects
        all_gt = all_gt.union(extra_tags)
        total_match = 0
        matched_cand_indices, matched_ref_indices = set(), set()
        for ii, cand in enumerate(all_cand):
            for jj, ref in enumerate(all_gt):
                if cand == ref:
                    matched_cand_indices.add(ii)
                    # print(cand, ref)
                    total_match += 1
                    break

        if self.synonym_matching:
            for ii, cand in enumerate(all_cand):
                if ii not in matched_cand_indices:
                    for jj, ref in enumerate(all_gt):
                        if self.compute_synonyms_score(cand, ref) > 0.:
                            matched_cand_indices.add(ii)
                            # print(cand, ref)
                            total_match += 1
                            break
        
        remained_cands = [cand for i, cand in enumerate(all_cand) if i not in matched_cand_indices]
        remained_refs = [gt for j, gt in enumerate(all_gt) if j not in matched_ref_indices]
        cand_match = total_match
        #ref_match = total_match
        if self.soft_matching and len(remained_cands) > 0 and len(remained_refs) > 0:
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    remained_cands_features, remained_refs_features = encode_phrases(self.text_encoder, remained_cands, remained_refs, batch_size=4)
            sim_mat = remained_cands_features.dot(remained_refs_features.T)
            remained_cands_match = np.sum(np.max(sim_mat, axis=1))
            remained_refs_match = np.sum(np.max(sim_mat, axis=0))
            cand_match = total_match + remained_cands_match
            #ref_match = total_match + remained_refs_match
        object_precision = cand_match / (len(cand_objects) + self.eps)

        return object_precision

    def compute_score(self, gts, res, extra_tags, prev_gt_parsed=None, prev_cand_parsed=None, return_parse_results=False, return_recalls=False):
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
        recalls = []
        precisions = []
        parse_results = []
        info_ratios = []
        for sample_key in tqdm.tqdm(gt_parsed.keys(), desc="computing score"):
            sample_gt_parsed, sample_cand_parsed = [sublist[:3] for sublist in gt_parsed[sample_key]], cand_parsed[sample_key][0][:3]
            sample_extra_tags = extra_tags[sample_key]

            sentence = res[sample_key][0]
            relations = cand_parsed[sample_key][0][3]
            attributes = cand_parsed[sample_key][0][4]
            objects = cand_parsed[sample_key][0][5]
            sample_info_ratio = self.remove_relations_from_sentence(sentence, relations, attributes, objects)
            info_ratios.append(sample_info_ratio)

            sample_precisions = [
                self.compute_precision(this_gt_parsed, sample_cand_parsed, sample_extra_tags) for this_gt_parsed in sample_gt_parsed
            ]

            results = [
                self.compute_f_score(this_gt_parsed, sample_cand_parsed) for this_gt_parsed in sample_gt_parsed
            ]
            sample_scores = [result[0] for result in results]
            sample_score = sum(sample_scores) / len(sample_scores)
            scores.append(sample_score)
            sample_recalls = [result[10] for result in results]
            sample_recall = sum(sample_recalls) / len(sample_recalls)
            recalls.append(sample_recall)
            sample_precision = sum(sample_precisions) / len(sample_precisions)
            precisions.append(sample_precision)
            parse_results.append({
                "sample_key": sample_key,
                "gt_parsed": sample_gt_parsed, 
                "cand_parsed": sample_cand_parsed,
                #"object_precision": round(mean([result[1]*100 for result in results]), 2), 
                'object_recall': round(mean([result[2]*100 for result in results]), 2), 
                #'object_f1': round(mean([result[3]*100 for result in results]), 2),  
                #'attribute_precision': round(mean([result[4]*100 for result in results]), 2),  
                'attribute_recall': round(mean([result[5]*100 for result in results]), 2),  
                #'attribute_f1': round(mean([result[6]*100 for result in results]), 2), 
                #'relation_precision': round(mean([result[7]*100 for result in results]), 2),  
                'relation_recall': round(mean([result[8]*100 for result in results]), 2),  
                #'relation_f1': round(mean([result[9]*100 for result in results]), 2), 
                'avg_recall': round(mean([result[10]*100 for result in results]), 2),
                'sample_precision': sample_precision,
                'sample_info_ratio': sample_info_ratio

            })

        score = sum(scores) / len(scores)
        recall = sum(recalls) / len(recalls)
        precision = sum(precisions) / len(precisions)
        info_ratio = sum(info_ratios) / len(info_ratios)

        if return_parse_results:
            if return_recalls:
                return score, scores, precision, precisions, recall, recalls, info_ratio, info_ratios, parse_results, 
            else:
                return score, scores, parse_results
        else:
            return score, scores



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







