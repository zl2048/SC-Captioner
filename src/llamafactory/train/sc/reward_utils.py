'''
rewards.py: 这个文件存放一些capture指标相关的计算函数
'''
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import collections
import re
import difflib
import io
import contextlib
from factual_scene_graph.evaluation.soft_spice_evaluation import encode_phrases
import numpy as np


def merge_sentence_results(results, text_processor):
    # from IPython import embed; embed()
    objects, attributes, relations = set(), collections.defaultdict(set), set()
    relations_original, attributes_original, objects_original, attributes_ = set(), collections.defaultdict(set), set(), set()
    for result in results:
        for entity in result['entities']:
            objects_original.add(entity['head'])
            lemmatized_obj = text_processor.normalize_word(entity['head'], wordnet.NOUN)
            objects.add(lemmatized_obj)
            for attribute in entity['attributes']:
                attributes_original[entity['head']].add(attribute)
                attributes_.add(attribute)
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

    return objects, attributes, relations, relations_original, attributes_original, objects_original, attributes_


def get_revision(
    objects_1,
    objects_2,
    attributes_1,
    attributes_2,
    relations_1,
    relations_2,
    text_1,
    text_2, 
    text_encoder,
    stop_words    
):
    '''
    input: components of two sentences
    return: removed and added components in fact
    '''
    
    removed_objects = objects_1 - objects_2
    added_objects = objects_2 - objects_1

    removed_objects_cache = set()
    added_objects_cache = set()
    for removed_object in (removed_objects):
        #if any(removed_object in s for s in removed + replaced[0]):
        if removed_object not in text_2 and removed_object in text_1:
            removed_objects_cache.add(removed_object)
    for added_object in (added_objects):
        if added_object not in text_1 and added_object in text_2:
            added_objects_cache.add(added_object)
    removed_objects = removed_objects_cache
    added_objects = added_objects_cache
    
    removed_objects_list = list(removed_objects)
    added_objects_list = list(added_objects)
    if removed_objects_list and added_objects_list:
        with io.StringIO() as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                removed_objects_features, added_objects_features = encode_phrases(text_encoder, removed_objects_list, added_objects_list, batch_size=4)
        sim_mat = removed_objects_features.dot(added_objects_features.T)

        to_remove_from_removed = []
        to_remove_from_added = []

        for i in range(len(removed_objects_list)):
            for j in range(len(added_objects_list)):
                if sim_mat[i, j] > 0.75:
                    to_remove_from_removed.append(i)
                    to_remove_from_added.append(j)

        removed_objects_list = [item for idx, item in enumerate(removed_objects_list) if idx not in to_remove_from_removed]
        added_objects_list = [item for idx, item in enumerate(added_objects_list) if idx not in to_remove_from_added]

    removed_relations = relations_1 - relations_2
    added_relations = relations_2 - relations_1

    removed_attributes = collections.defaultdict(set)
    added_attributes = collections.defaultdict(set)

    all_keys = set(attributes_1.keys()) | set(attributes_2.keys())

    for key in all_keys:
        values1 = attributes_1.get(key, set())
        values2 = attributes_2.get(key, set())

        only_in_1 = values1 - values2
        only_in_2 = values2 - values1

        if only_in_1:
            removed_attributes[key].update(only_in_1)
        if only_in_2:
            added_attributes[key].update(only_in_2)
    
    removed_attributes_cache = collections.defaultdict(set)
    added_attributes_cache = collections.defaultdict(set)
    for key in removed_attributes:
        if key in removed_objects:
            continue
        for attribute in removed_attributes[key]:
            if attribute not in text_2 and attribute in text_1:
                removed_attributes_cache[key].add(attribute)
    
    for key in added_attributes:
        if key in added_objects:
            continue
        for attribute in added_attributes[key]:
            if attribute not in text_1 and attribute in text_2:
                added_attributes_cache[key].add(attribute)
    removed_attributes = removed_attributes_cache
    added_attributes = added_attributes_cache

    removed_objects = set(removed_objects_list)
    added_objects = set(added_objects_list)
    
    if stop_words:
        from capture_metric.stop_words import stop_words_list
        stop_words_list = set(stop_words_list)
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()

        removed_objects_cache = set()
        added_objects_cache = set()
        for word in removed_objects:
            singular_word = wnl.lemmatize(word, pos='n')
            if singular_word not in stop_words_list:
                removed_objects_cache.add(word)
        for word in added_objects:
            singular_word = wnl.lemmatize(word, pos='n')
            if singular_word not in stop_words_list:
                added_objects_cache.add(word)

        removed_objects = removed_objects_cache
        added_objects = added_objects_cache
        

    return removed_objects, added_objects, removed_relations, added_relations, removed_attributes, added_attributes

