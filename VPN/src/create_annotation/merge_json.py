import json
import random
import sys
random.seed(0)

def main():
    with open('./output_json/CPC_label_500.json') as f:
        json_str1 = f.read()
    dict1 = json.loads(json_str1) # list

    with open('./output_json/AADB_label_500.json') as f:
        json_str2 = f.read()
    dict2 = json.loads(json_str2) # list
    dict_new = dict1 + dict2

    with open('./output_json/CPC_AADB_label_500.json', 'w') as f:
        json.dump(dict_new, f, indent=2)
    print('ok')


if __name__ == '__main__':
    main()
