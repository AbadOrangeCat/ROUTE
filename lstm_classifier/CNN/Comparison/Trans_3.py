# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# ==========================================================
# 1. 数据加载与预处理 (Data Loading and Preprocessing)
# ==========================================================
# 使用原始 fake.csv 与 true.csv 数据集
fake_df = pd.read_csv('../../news/fake.csv', encoding='utf-8')
real_df = pd.read_csv('../../news/true.csv', encoding='utf-8')

covid_fake_df = pd.read_csv('../../covid/fakeNews.csv', encoding='utf-8')
covid_real_df = pd.read_csv('../../covid/trueNews.csv', encoding='utf-8')

# 筛选“政治”文本 (subject 字段)
politics_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
politics_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']

# 筛选“医疗”文本 (subject 字段)
medical_fake = covid_fake_df[(covid_fake_df['Text'].str.len() >= 40)]['Text']
medical_real = covid_real_df[(covid_real_df['Text'].str.len() >= 40)]['Text']

# 合并并标注 (0 = 假新闻, 1 = 真新闻)
policy_texts = pd.concat([politics_fake, politics_real]).reset_index(drop=True)
policy_labels = np.concatenate([
    np.zeros(len(politics_fake), dtype=int),
    np.ones(len(politics_real), dtype=int)
])
medical_texts = pd.concat([medical_fake, medical_real]).reset_index(drop=True)
medical_labels = np.concatenate([
    np.zeros(len(medical_fake), dtype=int),
    np.ones(len(medical_real), dtype=int)
])

# 将文本列表赋给训练/测试变量
X_train_texts = policy_texts.tolist()
y_train = policy_labels
X_test_texts  = medical_texts.tolist()
y_test  = medical_labels
# 定义词汇表大小和最大序列长度
vocab_size = 20000  # 只考虑最常见的 20000 个词
maxlen = 200        # 每条新闻截断或填充到 200 个单词

# 使用 Tokenizer 将文本转换为序列
tokenizer = Tokenizer(num_words=vocab_size, oov_token="[UNK]")
tokenizer.fit_on_texts(X_train_texts)
X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

# 将序列填充/截断到固定长度
X_train = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
X_test = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

# 2. 构建 Transformer 模型并在政治新闻数据上训练 (Model Definition and Training)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        # 多头自注意力
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差连接 + LayerNorm
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen_tensor = tf.shape(x)[-1]  # 序列实际长度（动态获取）
        positions = tf.range(start=0, limit=maxlen_tensor, delta=1)
        pos_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + pos_embeddings

def create_transformer_model(maxlen, vocab_size, embed_dim=32, num_heads=2, ff_dim=32):
    inputs = layers.Input(shape=(maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)          # 序列池化为定长向量
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid输出真假概率
    model = Model(inputs, outputs)
    return model

model = create_transformer_model(maxlen, vocab_size, embed_dim=32, num_heads=2, ff_dim=64)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 在政治新闻训练集上训练模型
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)



y_pred_prob = model.predict(X_test)
y_pred_prob = y_pred_prob.flatten()
pred_labels_num = []
pred_categories = []
for p in y_pred_prob:
    if p > 0.000131:
        pred_labels_num.append(1)
        pred_categories.append("True")
    elif p < 0.00004:
        pred_labels_num.append(0)
        pred_categories.append("False")
    else:
        pred_labels_num.append(None)
        pred_categories.append("Unsure")

# 4. 保存预测结果到 CSV 文件 (Save Predictions to CSV)
results_df = pd.DataFrame({
    "text": X_test_texts,
    "pred_probability": y_pred_prob,
    "pred_category": pred_categories
})
results_df.to_csv("predictions.csv", index=False, encoding='utf-8')

# 5. 计算评估指标并输出 (Evaluation Metrics)
# 总体准确率（将“不确定”视为错误预测）
total_samples = len(y_test)
correct_incl_uncertain = 0
for true_label, pred_label in zip(y_test, pred_labels_num):
    if pred_label is not None and pred_label == true_label:
        correct_incl_uncertain += 1
overall_accuracy = correct_incl_uncertain / total_samples if total_samples > 0 else 0.0

# 分别计算真新闻和假新闻的精确率、召回率、F1分数
tp_real = fp_real = fn_real = 0
tp_fake = fp_fake = fn_fake = 0
for true_label, pred_label in zip(y_test, pred_labels_num):
    # 真新闻（正类=1）
    if pred_label == 1 and true_label == 1:
        tp_real += 1
    if pred_label == 1 and true_label == 0:
        fp_real += 1
    if true_label == 1 and pred_label != 1:
        fn_real += 1
    # 假新闻（正类=0）
    if pred_label == 0 and true_label == 0:
        tp_fake += 1
    if pred_label == 0 and true_label == 1:
        fp_fake += 1
    if true_label == 0 and pred_label != 0:
        fn_fake += 1

precision_real = tp_real / (tp_real + fp_real) if (tp_real + fp_real) > 0 else 0.0
recall_real = tp_real / (tp_real + fn_real) if (tp_real + fn_real) > 0 else 0.0
f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0.0

precision_fake = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0.0
recall_fake = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0
f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0.0

# 排除“不确定”结果后的准确率
definite_total = sum(1 for label in pred_labels_num if label is not None)
definite_correct = sum(1 for true_label, pred_label in zip(y_test, pred_labels_num) if pred_label is not None and pred_label == true_label)
accuracy_excluding_uncertain = definite_correct / definite_total if definite_total > 0 else 0.0

# 输出指标
print(f"总体准确率 (不确定视为错误): {overall_accuracy * 100:.2f}%")
print(f"真新闻 - 精确率: {precision_real * 100:.2f}%, 召回率: {recall_real * 100:.2f}%, F1: {f1_real * 100:.2f}%")
print(f"假新闻 - 精确率: {precision_fake * 100:.2f}%, 召回率: {recall_fake * 100:.2f}%, F1: {f1_fake * 100:.2f}%")
print(f"排除不确定结果后的准确率: {accuracy_excluding_uncertain * 100:.2f}%")

# 6. 绘制可选图表 (Visualization - Optional)
# 每类预测结果的数量条形图
category_counts = {
    "真新闻": pred_categories.count("真新闻"),
    "假新闻": pred_categories.count("假新闻"),
    "不确定": pred_categories.count("不确定")
}
plt.figure()
plt.bar(category_counts.keys(), category_counts.values(), color=['green', 'red', 'gray'])
plt.title("各类别预测结果数量")
plt.xlabel("预测类别")
plt.ylabel("新闻数量")
plt.savefig("category_counts.png")

# 模型预测为“真新闻”的概率分布直方图
real_probs = [p for p, cat in zip(y_pred_prob, pred_categories) if cat == "真新闻"]
plt.figure()
plt.hist(real_probs, bins=10, range=(0.55, 1.0), color='green', edgecolor='black')
plt.title("预测为真新闻的概率分布")
plt.xlabel("预测为真新闻的概率")
plt.ylabel("频数")
plt.savefig("real_news_prob_dist.png")

print("图表已保存：category_counts.png，real_news_prob_dist.png")

df = pd.DataFrame({
    "true_label": y_test,
    "pred_category": pred_categories
})

total = len(df)

def summarize_category(df, category):
    count = (df["pred_category"] == category).sum()
    pct = count / total * 100
    if category in ["真新闻", "假新闻"]:
        # 确定正类标签
        true_val = 1 if category == "真新闻" else 0
        # 真阳性
        tp = ((df["pred_category"] == category) & (df["true_label"] == true_val)).sum()
        # 预测为该类的总数（用于精确率）
        pp = count
        # 实际该类的总数（用于召回率）
        ap = (df["true_label"] == true_val).sum()
        precision = tp / pp * 100 if pp > 0 else 0.0
        recall = tp / ap * 100 if ap > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print("pctpct",category,precision)
        return {
            "数量": count,
            "百分比": f"{pct:.2f}%",
            "精确率": f"{precision:.2f}%",
            "召回率": f"{recall:.2f}%",
            "F1":       f"{f1:.2f}%"
        }
    else:
        # 不确定类只输出数量和占比
        return {
            "数量": count,
            "百分比": f"{pct:.2f}%",
            "精确率": "--",
            "召回率": "--",
            "F1":       "--"
        }

# 对三类分别统计
rows = []
for cat in ["真新闻", "假新闻", "不确定"]:
    stats = summarize_category(df, cat)
    stats["类别"] = cat
    rows.append(stats)

# 构造 DataFrame 并输出
stats_df = pd.DataFrame(rows, columns=["类别", "数量", "百分比", "精确率", "召回率", "F1"])
print(stats_df.to_string(index=False, justify="left"))

