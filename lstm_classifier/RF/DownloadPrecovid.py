from datasets import load_dataset
import pandas as pd
import os

# 1. 下载/加载 PubHealth 数据集
print("正在从 Hugging Face 下载 PubHealth 数据集...")
# 注意：第一次运行会自动下载并缓存，速度取决于网络
dataset = load_dataset("bigbio/pubhealth", "pubhealth_source")

# 2. 直接保存原始分片 (Train, Test, Validation)
print("下载完成，正在保存为 CSV 文件...")

# 循环处理每一个分片（train, test, validation）
for split in dataset.keys():
    # 转换为 Pandas DataFrame
    df = dataset[split].to_pandas()

    # 定义文件名，例如 pubhealth_train.csv
    filename = f"pubhealth_{split}.csv"

    # 保存
    df.to_csv(filename, index=False)
    print(f"✅ 已保存: {filename} (包含 {len(df)} 条数据)")

print("-" * 30)
print("🎉 所有文件已保存在当前目录下。")