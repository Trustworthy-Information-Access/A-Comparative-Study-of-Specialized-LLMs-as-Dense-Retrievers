import csv
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Add a new column with value 0 to the second column of a TSV file')
parser.add_argument('data_name', help='Name of the data file')
args = parser.parse_args()

data_name = args.data_name

# 读取原始tsv文件
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/qrels/test.tsv', 'r') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    data = list(tsv_reader)

# 在每行的第二列插入新列数值为0
for row in data:
    row.insert(1, 'Q0')

# 写入更新后的tsv文件，不包含表头
with open(f'/root/paddlejob/workspace/env_run/data/beir/BEIR/{data_name}/qrels/test_qrels.tsv', 'w', newline='') as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    tsv_writer.writerows(data[1:])  # 从第二行开始写入，不包含表头