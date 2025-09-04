'''
change the image path in data json
'''

import json
import os

def update_image_paths(json_file, new_directory):
    new_directory = os.path.normpath(new_directory)  #
    if not new_directory.endswith(os.sep):
        new_directory += os.sep

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if "images" in item and isinstance(item["images"], list) and len(item["images"]) > 0:
            old_path = item["images"][0]
            filename = os.path.basename(old_path)
            new_path = new_directory + filename
            item["images"][0] = new_path

    output_file = json_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"file saved to: {output_file}")

if __name__ == "__main__":
    json_file_path = "test_docci500.json"        
    image_directory = "/your/new/directory" 

    update_image_paths(json_file_path, image_directory)