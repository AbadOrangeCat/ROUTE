# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import csv
import matplotlib.pyplot as plt

# ==========================================================
# 1. 数据加载与预处理 (Data Loading and Preprocessing)
# ==========================================================

needTrain = False

fake_df = pd.read_csv('../../news/fake.csv', encoding='utf-8')
real_df = pd.read_csv('../../news/true.csv', encoding='utf-8')
covid_fake_df = pd.read_csv('../../covid/fakeNews.csv', encoding='utf-8')
covid_real_df = pd.read_csv('../../covid/trueNews.csv', encoding='utf-8')

politics_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
politics_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']
medical_fake = covid_fake_df[covid_fake_df['Text'].str.len() >= 40]['Text']
medical_real = covid_real_df[covid_real_df['Text'].str.len() >= 40]['Text']

policy_texts = pd.concat([politics_fake, politics_real]).reset_index(drop=True)
policy_labels = np.concatenate([np.zeros(len(politics_fake), dtype=int), np.ones(len(politics_real), dtype=int)])
medical_texts = pd.concat([medical_fake, medical_real]).reset_index(drop=True)
medical_labels = np.concatenate([np.zeros(len(medical_fake), dtype=int), np.ones(len(medical_real), dtype=int)])

X_train_texts, y_train = policy_texts.tolist(), policy_labels
X_test_texts, y_test   = medical_texts.tolist(), medical_labels
vocab_size, maxlen = 20000, 200
print(f"训练集样本数: {len(X_train_texts)},  真新闻比例: {y_train.mean():.2f}")
print(f"测试集样本数: {len(X_test_texts)},  真新闻比例: {y_test.mean():.2f}")

tokenizer = Tokenizer(num_words=vocab_size, oov_token="[UNK]")
tokenizer.fit_on_texts(X_train_texts)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=maxlen, padding='post', truncating='post')
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test_texts),  maxlen=maxlen, padding='post', truncating='post')

# 2. 构建并训练 Transformer 模型
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs)
        attn = self.drop1(attn, training=training)
        out1 = self.norm1(inputs + attn)
        ffn = self.ffn(out1)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(out1 + ffn)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb   = layers.Embedding(maxlen, embed_dim)
    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(seq_len)
        return self.token_emb(x) + self.pos_emb(positions)

def create_model():
    inp = layers.Input((maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, 32)(inp)
    x = TransformerBlock(32, 2, 64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out)

if(needTrain):
    model = create_model()
    model.compile('adam', 'binary_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)
    model.save('saved_transformer_model')
else:
    model = load_model('saved_transformer_model')

y_pred_prob = model.predict(X_test).flatten()
# y_pred_prob = model.predict(X_train).flatten()
# y_test = y_train

plt.figure(figsize=(8, 4))
plt.hist(y_pred_prob, bins=50, edgecolor='black')
plt.xlabel('Predicted probability $p$')
plt.ylabel('Frequency')
plt.title('Distribution of $p$ (model output)')
plt.tight_layout()
plt.show()


pTrue = 0.0007
pFlase = 0.0000045
derate = 1
data = [
    ['True News', 'Count', 'Percentage','Precision','Recall','F1','Accuracy','Fake News', 'Count', 'Percentage','Precision','Recall','F1','Accuracy','Unknow', 'Count', 'Percentage'],
]
while pFlase>=0:
    pred_labels, pred_cats = [], []
    for p in y_pred_prob:
        if p > pTrue:
            pred_labels.append(1); pred_cats.append("真新闻")
        if p < pFlase:
            pred_labels.append(0); pred_cats.append("假新闻")
        if p > pFlase and p < pTrue:
            pred_labels.append(None); pred_cats.append("不确定")

    # 3. 评估指标计算
    # 计算各类 TP, FP, FN
    tp_real = sum(1 for t, p in zip(y_test, pred_labels) if p == 1 and t == 1)
    fp_real = sum(1 for t, p in zip(y_test, pred_labels) if p == 1 and t == 0)
    fn_real = sum(1 for t, p in zip(y_test, pred_labels) if p == 0 and t == 1)

    tp_fake = sum(1 for t, p in zip(y_test, pred_labels) if p == 0 and t == 0)
    fp_fake = sum(1 for t, p in zip(y_test, pred_labels) if p == 0 and t == 1)
    fn_fake = sum(1 for t, p in zip(y_test, pred_labels) if p == 1 and t == 0)

    # 计算精确率、召回率、F1

    # def calc_metrics(tp, fp, fn):
    #     precision = tp / (tp + fp) * 100 if tp + fp else 0.0
    #     recall    = tp / (tp + fn) * 100 if tp + fn else 0.0
    #     f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    #     return precision, recall, f1
    #

    def calc_metrics(tp, fp, fn):
        """Return precision, recall, f1 **and** accuracy (all as percentages)."""
        precision = tp / (tp + fp) * 100 if tp + fp else 0.0
        recall = tp / (tp + fn) * 100 if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = tp / (tp + fp ) * 100 if tp + fp else 0.0
        return precision, recall, f1, accuracy

    prec_real, rec_real, f1_real,acc_real = calc_metrics(tp_real, fp_real, fn_real)
    prec_fake, rec_fake, f1_fake,acc_fake = calc_metrics(tp_fake, fp_fake, fn_fake)

    print("sss",tp_real, fp_real, fn_real,acc_real)

    # 计算各类准确率 (分类准确率 = TP/实际样本数)
    ap_real = sum(1 for t in y_test if t == 1)
    ap_fake = sum(1 for t in y_test if t == 0)


    # 构建统计表格
    total = len(y_test)
    rows = []
    for name, metrics, acc in [("真新闻", (prec_real, rec_real, f1_real), acc_real),
                               ("假新闻", (prec_fake, rec_fake, f1_fake), acc_fake),
                               ("不确定", None, None)]:
        count = pred_cats.count(name)
        pct   = count / total * 100
        row = {"类别": name, "数量": count, "百分比": f"{pct:.2f}%"}
        if metrics:
            row.update({
                "精确率": f"{metrics[0]:.2f}%",
                "召回率": f"{metrics[1]:.2f}%",
                "F1":      f"{metrics[2]:.2f}%",
                "准确率":  f"{acc:.2f}%"
            })
        else:
            row.update({"精确率": "--", "召回率": "--", "F1": "--", "准确率": "--"})
        rows.append(row)

    stats_df = pd.DataFrame(rows, columns=["类别", "数量", "百分比", "精确率", "召回率", "F1", "准确率"])
    # print(stats_df.to_string(index=False, justify="left"))
    print(pTrue,pFlase)
    print()
    saveData = []
    for row in rows:
        saveData.extend(row.values())
    saveData[0] = pTrue
    saveData[7] = pFlase
    data.append(saveData)
    # pTrue -= derate
    pFlase -= derate

with open('False0001-000.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)