import csv
import argparse
import json
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Add a new column with value 0 to the second column of a TSV file')
parser.add_argument('data_name', help='Name of the data file')
args = parser.parse_args()

data_name = args.data_name

# 读取原始tsv文件
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/qrels/test.tsv', 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    data = list(tsv_reader)
query_ids = [i[0] for i in data][1:]
total_data_line = []
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/queries.jsonl', 'r') as file:
    for line in file:
        js = json.loads(line)
        if js["_id"] in query_ids:
            total_data_line.append(js)


# 写入更新后的tsv文件，不包含表头
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/queries.jsonl', 'w', newline='') as file:
    for data in total_data_line:
        file.write(json.dumps(data)+"\n")