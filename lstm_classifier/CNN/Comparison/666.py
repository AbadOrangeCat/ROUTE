# -*- coding: utf-8 -*-
"""
News classification with Transformer and saving results
======================================================
This script trains (or loads) a lightweight Transformer text‑classifier and
applies it to two datasets (politics & medical) that have been
pre‑labelled as fake / real. After inference the samples are split into
three categories – 真新闻 (true), 假新闻 (fake) and 不确定 (unknown) – according
to a pair of probability thresholds, and each category is written to its
own CSV file.

**Change log 2025‑07‑14**
─────────────────────────
* Replaced `pred_labels: list[int | None]` with `List[Optional[int]]` for
  compatibility with Python ≤ 3.9 (PEP 604 union syntax `|` requires
  Python 3.10 or newer).
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from typing import List, Optional  # <-- added for older Python versions
import re
# ==========================================================
# 1. 数据加载与预处理 (Data Loading and Preprocessing)
# ==========================================================

need_train = False  # change to True if you want to re‑train from scratch

# source files -------------------------------------------------------------
fake_df        = pd.read_csv('../../news/fake.csv',       encoding='utf-8')
real_df        = pd.read_csv('../../news/true.csv',       encoding='utf-8')
# covid_fake_df  = pd.read_csv('../../covid/fakeNews.csv',  encoding='utf-8')
# covid_real_df  = pd.read_csv('../../covid/trueNews.csv',  encoding='utf-8')

covid_fake_df  = pd.read_csv('../../news/fake.csv',  encoding='utf-8')
covid_real_df  = pd.read_csv('../../news/true.csv',  encoding='utf-8')

url_pat = re.compile(r'https?://\S+|www\.\S+')  # 匹配 http(s):// 或 www.

def strip_urls(series: pd.Series) -> pd.Series:
    """把一整列文本里的 URL 去掉"""
    return series.astype(str).str.replace(url_pat, '', regex=True)

# 读完 CSV 立刻批量清洗
# fake_df['text']        = strip_urls(fake_df['text'])
# real_df['text']        = strip_urls(real_df['text'])
# covid_fake_df['text']  = strip_urls(covid_fake_df['text'])
# covid_real_df['text']  = strip_urls(covid_real_df['text'])

# select politics & medical subsets ---------------------------------------
politics_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
politics_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']
# medical_fake  = covid_fake_df[covid_fake_df['Text'].str.len() >= 40]['Text']
# medical_real  = covid_real_df[covid_real_df['Text'].str.len() >= 40]['Text']

medical_fake  = covid_fake_df[(covid_fake_df['subject'] == 'News') & (covid_fake_df['text'].str.len() >= 40)]['text']
medical_real  = covid_real_df[(covid_real_df['subject'] == 'worldnews') & (covid_real_df['text'].str.len() >= 40)]['text']

policy_texts  = pd.concat([politics_fake, politics_real]).reset_index(drop=True)
policy_labels = np.concatenate([np.zeros(len(politics_fake), dtype=int),
                                np.ones(len(politics_real),  dtype=int)])
medical_texts  = pd.concat([medical_fake, medical_real]).reset_index(drop=True)
medical_labels = np.concatenate([np.zeros(len(medical_fake), dtype=int),
                                 np.ones(len(medical_real),  dtype=int)])

X_train_texts, y_train = policy_texts.tolist(), policy_labels
X_test_texts,  y_test  = medical_texts.tolist(), medical_labels

vocab_size, maxlen = 20_000, 200
print(f"训练集样本数: {len(X_train_texts)},  真新闻比例: {y_train.mean():.2f}")
print(f"测试集样本数: {len(X_test_texts)},  真新闻比例: {y_test.mean():.2f}")

# tokenise & pad -----------------------------------------------------------

tokenizer = Tokenizer(num_words=vocab_size, oov_token="[UNK]")
tokenizer.fit_on_texts(X_train_texts)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts),
                        maxlen=maxlen, padding='post', truncating='post')
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test_texts),
                        maxlen=maxlen, padding='post', truncating='post')

# ==========================================================
# 2. 构建/加载 Transformer 模型 (Build / Load Transformer Model)
# ==========================================================

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        self.att  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn  = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training: bool = False):
        attn = self.att(inputs, inputs)
        attn = self.drop1(attn, training=training)
        out1 = self.norm1(inputs + attn)
        ffn  = self.ffn(out1)
        ffn  = self.drop2(ffn, training=training)
        return self.norm2(out1 + ffn)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb   = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        seq_len  = tf.shape(x)[-1]
        positions = tf.range(seq_len)
        return self.token_emb(x) + self.pos_emb(positions)

def create_model():
    inputs = layers.Input((maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, 32)(inputs)
    x = TransformerBlock(32, 2, 64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

if need_train:
    model = create_model()
    model.compile('adam', 'binary_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)
    model.save('saved_transformer_model')
else:
    model = load_model('saved_transformer_model')

# ==========================================================
# 3. 推理 (Inference)
# ==========================================================

y_pred_prob = model.predict(X_test, batch_size=512).flatten()

# 可视化概率分布 -----------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(y_pred_prob, bins=50, edgecolor='black')
plt.xlabel('Predicted probability p')
plt.ylabel('Frequency')
plt.title('Distribution of p (model output)')
plt.tight_layout()
plt.show()

# ==========================================================
# 4. 分类阈值与预测标签 (Thresholds and Predicted Labels)
# ==========================================================

p_true  = 0.0007      # 上阈值: > p_true  -> 真新闻
p_false = 0.0000045   # 下阈值: < p_false -> 假新闻

pred_labels: List[Optional[int]] = []  # type‑safe for <=3.9
pred_cats:   List[str] = []

for p in y_pred_prob:
    if p > p_true:
        pred_labels.append(1)
        pred_cats.append("真新闻")
    elif p < p_false:
        pred_labels.append(0)
        pred_cats.append("假新闻")
    else:
        pred_labels.append(None)
        pred_cats.append("不确定")

# ==========================================================
# 5. 将分类结果写入三个 CSV 文件 (Save to 3 separate CSV files)
# ==========================================================

results_df = pd.DataFrame({
    'text':        X_test_texts,
    'true_label':  y_test,
    'pred_prob':   y_pred_prob*100,
    'pred_label':  pred_labels,
    'pred_cat':    pred_cats
})

results_df[results_df['pred_cat'] == '真新闻'].to_csv('true_news.csv',    index=False, encoding='utf-8')
results_df[results_df['pred_cat'] == '假新闻'].to_csv('fake_news.csv',    index=False, encoding='utf-8')
results_df[results_df['pred_cat'] == '不确定'].to_csv('unknown_news.csv', index=False, encoding='utf-8')

print("分类结果已保存为: true_news.csv, fake_news.csv, unknown_news.csv")

# ==========================================================
# 6. 评估指标 (Evaluation Metrics)
# ==========================================================

def calc_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) * 100 if tp + fp else 0.0
    recall    = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy  = tp / (tp + fp) * 100 if tp + fp else 0.0
    return precision, recall, f1, accuracy

# convert None -> -1 for easier comparison
clean_pred_labels = [lbl if lbl is not None else -1 for lbl in pred_labels]

tp_real = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 1 and t == 1)
fp_real = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 1 and t == 0)
fn_real = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 0 and t == 1)

tp_fake = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 0 and t == 0)
fp_fake = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 0 and t == 1)
fn_fake = sum(1 for t, p in zip(y_test, clean_pred_labels) if p == 1 and t == 0)

prec_real, rec_real, f1_real, acc_real = calc_metrics(tp_real, fp_real, fn_real)
prec_fake, rec_fake, f1_fake, acc_fake = calc_metrics(tp_fake, fp_fake, fn_fake)

print("\n评估指标: (单位 %)")
print(f"真新闻  精确率: {prec_real:.2f}  召回率: {rec_real:.2f}  F1: {f1_real:.2f}  准确率: {acc_real:.2f}")
print(f"假新闻  精确率: {prec_fake:.2f}  召回率: {rec_fake:.2f}  F1: {f1_fake:.2f}  准确率: {acc_fake:.2f}")
