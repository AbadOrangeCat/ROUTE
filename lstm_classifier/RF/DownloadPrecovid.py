from datasets import load_dataset
import pandas as pd
import os

print("正在从 Hugging Face 下载 HumAID 数据集...")

# ✅ 关键改动：关闭校验，避免 ExpectedMoreSplitsError
try:
    dataset = load_dataset("QCRI/HumAID-all", verification_mode="no_checks")
except TypeError:
    # 兼容旧版本 datasets（没有 verification_mode 参数）
    dataset = load_dataset("QCRI/HumAID-all", ignore_verifications=True)

print("下载完成，正在保存为 CSV 文件...")

# 建议保存到单独目录，避免污染工程目录
out_dir = "humaid_csv"
os.makedirs(out_dir, exist_ok=True)

print(dataset.keys())  # 一般会是: dict_keys(['train','validation','test'])

for split in dataset.keys():
    df = dataset[split].to_pandas()
    filename = os.path.join(out_dir, f"{split}.csv")
    df.to_csv(filename, index=False, encoding="utf-8-sig")  # utf-8-sig 方便 Excel 打开
    print(f"✅ 已保存: {filename} (包含 {len(df)} 条数据)")

print("-" * 30)
print("🎉 所有文件已保存在:", os.path.abspath(out_dir))
