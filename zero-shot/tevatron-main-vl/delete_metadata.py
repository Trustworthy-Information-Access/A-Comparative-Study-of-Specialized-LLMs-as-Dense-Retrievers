import csv
import argparse
import json

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Add a new column with value 0 to the second column of a TSV file')
parser.add_argument('data_name', help='Name of the data file')
args = parser.parse_args()

data_name = args.data_name

# 读取原始tsv文件
data_lines = []
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/queries.jsonl', 'r') as file:
    for line in file:
        js = json.loads(line)
        js["metadata"] = {}
        data_lines.append(js)
    

# 写入更新后的tsv文件，不包含表头
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/queries.jsonl', 'w', newline='') as file:
    for line in data_lines:
        file.write(json.dumps(line)+"\n")