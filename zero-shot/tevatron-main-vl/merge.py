import argparse
import glob
import pickle
import re
import numpy as np
import os

def merge_query_embeddings(input_pattern, output_path):

    # 获取并排序输入文件
    file_list = sorted(glob.glob(input_pattern), 
                    key=lambda x: int(x.split('.')[-2]))

    # 验证找到的文件
    print(f"找到 {len(file_list)} 个待合并文件：")
    for f in file_list:
        print(f" - {os.path.basename(f)}")

    # 初始化存储结构
    merged_arrays = []

    # 合并处理逻辑
    try:
        # 遍历加载并合并数据
        for file_path in file_list:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                
                # 验证数据结构
                if not isinstance(data, tuple) or len(data) < 1:
                    raise ValueError(f"文件 {file_path} 数据结构不符合要求")
                    
                array_data = data[0]
                if not isinstance(array_data, np.ndarray):
                    raise ValueError(f"文件 {file_path} 包含非NumPy数组数据")
                    
                merged_arrays.append(array_data)
                print(f"已加载 {array_data.shape} 形状的数组")

        # 合并数组（假设按第一个维度堆叠）
        final_array = np.concatenate(merged_arrays, axis=0)
        print(f"\n最终合并数组形状：{final_array.shape}")

        # 保存合并结果（保持元组格式）
        with open(output_path, "wb") as f:
            pickle.dump((final_array,), f)
        print(f"\n合并完成，结果已保存至：{output_path}")

    except Exception as e:
        print(f"\n合并过程中发生错误：{str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并query embedding文件')
    parser.add_argument('--input_pattern', type=str, 
                        default="query_emb.*.pkl",
                        help='./data/query_emb.*.pkl')
    parser.add_argument('--output_path', type=str,
                        default="query_emb.pkl",
                        help='./output/query_emb.pkl')
    
    args = parser.parse_args()
    
    try:
        merge_query_embeddings(args.input_pattern, args.output_path)
    except Exception as e:
        print(f"合并失败: {str(e)}")
        exit(1)