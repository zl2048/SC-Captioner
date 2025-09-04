'''
这个代码只计算object和attributes的precision和recall
'''
from capture import CAPTURE
import json
import os,sys
import pickle

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python xxx.py <inference output folder>")
        sys.exit(1)
    lf_folder = sys.argv[1]
    lf_path = os.path.join(lf_folder, "generated_predictions.jsonl")
    gt_path = "evaluate_docci500/test_docci_500_gt.json"
    cand_parsed_path = os.path.join(lf_folder, "cand_parsed_ours.pkl")
    with open(gt_path, 'r', encoding='utf-8') as file:
        gt_dict = json.load(file)
    refs = {}
    preds = {}
    extra_tags = {}

    with open(lf_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            gt_caption = item['label']
            for key, value in gt_dict.items():
                if value == gt_caption:
                    image_name = key
            refs[image_name] = [gt_caption]
            preds[image_name] = [item['predict']]
    
    # 读取额外的objects和attributes
    extra_objects = {}
    with open("evaluate_docci500/docci500_gpt_objects.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            key = next(iter(item))
            extra_objects[key] = item[key]
    with open("evaluate_docci500/docci500_gpt_attributes.json", "r") as f:
        extra_attributes = json.load(f)

    # 将parse好的gt和cand直接读入
    if os.path.isfile(cand_parsed_path):
        with open(cand_parsed_path, 'rb') as file: 
            prev_cand_parsed = pickle.load(file)
    else:
        prev_cand_parsed = None
        
    if os.path.isfile("evaluate_docci500/gt_parsed_ours.pkl"):
        with open('evaluate_docci500/gt_parsed_ours.pkl', 'rb') as file: 
            prev_gt_parsed = pickle.load(file)
    else:
        prev_gt_parsed = None
        
    evaluator = CAPTURE()
    object_precision, object_recall, object_f1, attribute_precision, attribute_recall, attribute_f1, cand_parsed = evaluator.compute_score(refs, preds, prev_gt_parsed=prev_gt_parsed, prev_cand_parsed=prev_cand_parsed, extra_objects=extra_objects, extra_attributes=extra_attributes, return_parse_results=True)
    with open(f"{lf_folder}/metrics.txt", "a") as f: 
        print(f"object precision turn1: {object_precision:.4f}", file=f)
        print(f"object recall turn1: {object_recall:.4f}", file=f)
        print(f"object f1 turn1: {object_f1:.4f}", file=f)
        print(f"attribute precision turn1: {attribute_precision:.4f}", file=f)
        print(f"attribute recall turn1: {attribute_recall:.4f}", file=f)
        print(f"attribute f1 turn1: {attribute_f1:.4f}", file=f)
    
    with open(cand_parsed_path, 'wb') as file: 
        pickle.dump(cand_parsed,file)


