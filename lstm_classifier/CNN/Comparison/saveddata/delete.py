import pandas as pd
import csv   # 用来控制 to_csv 的 quoting 行为

# 1⃣ 把整列当成字符串读进来，确保能拿到引号
df = pd.read_csv('tt.csv', dtype={'pred_prob': str})

# 2⃣ 去掉各种可能的前缀符号（普通 '、中文引号 ’‘ 以及空格、制表符）
df['pred_prob'] = (
    df['pred_prob']
      .str.replace(r"^[\s\ufeff]*['’‘]", "", regex=True)   # 只去掉最左边的引号/空白/BOM
)

# 3⃣ 转回浮点数；errors='coerce' 可把非数字强制变 NaN，避免因脏数据报错
df['pred_prob'] = pd.to_numeric(df['pred_prob'], errors='coerce')

# 4⃣ 保存：保证列是真正的 float，且不对数值加额外引号
df.to_csv(
    'tt3.csv',
    index=False,
    float_format="%.10g",        # 控制科学计数法/小数位展示，可按需修改
    quoting=csv.QUOTE_MINIMAL    # 确保数值列不被强制加引号
)
