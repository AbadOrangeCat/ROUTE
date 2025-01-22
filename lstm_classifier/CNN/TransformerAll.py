import sys
import tensorflow as tf
# 首先确保 tf.keras 可用
from tensorflow import keras as tfkeras

# 尝试导入独立keras，如果没有则将tf.keras映射为keras
try:
    import keras
except ImportError:
    # 如果keras没安装，则通过mock将tf.keras当做keras使用
    sys.modules['keras'] = tfkeras
    import keras

# 对独立 keras 进行猴子补丁
if not hasattr(keras.utils, "unpack_x_y_sample_weight"):
    def unpack_x_y_sample_weight(data):
        x, y = data
        return x, y, None
    keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# -------------------------------
# 数据加载和预处理
# -------------------------------
# 请根据自己的数据路径修改
fake_df = pd.read_csv('../news/fake.csv')  # 假政治新闻数据
real_df = pd.read_csv('../news/true.csv')  # 真政治新闻数据

covid_fake_df = pd.read_csv('../news/fake.csv')  # Fake medical news dataset
covid_real_df = pd.read_csv('../news/true.csv')  # Real medical news dataset


# 筛选政治新闻
filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]
real_texts = filtered_df['text']

# 筛选医疗新闻
filtered_df = fake_df[(fake_df['subject'] == 'News') & (fake_df['text'].str.len() >= 40)]
covid_fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'worldnews') & (real_df['text'].str.len() >= 40)]
covid_real_texts = filtered_df['text']

# 合并政治数据与标签：0为假，1为真
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([np.zeros(len(fake_texts)), np.ones(len(real_texts))])

# 合并医疗数据与标签：0为假，1为真
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([np.zeros(len(covid_fake_texts)), np.ones(len(covid_real_texts))])

# 数据集划分
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(policy_texts, policy_labels, test_size=0.2, random_state=42)
X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(medical_texts, medical_labels, test_size=0.2, random_state=42)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

maxlen = 256

def encode_texts(texts, tokenizer, max_length=256):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='tf')

# 编码政治数据
train_encodings_p = encode_texts(X_train_p, tokenizer, max_length=maxlen)
val_encodings_p = encode_texts(X_val_p, tokenizer, max_length=maxlen)

# 编码医疗数据
train_encodings_m_full = encode_texts(X_train_m_full, tokenizer, max_length=maxlen)
val_encodings_m = encode_texts(X_val_m, tokenizer, max_length=maxlen)

# 转换标签为张量
y_train_p_tf = tf.convert_to_tensor(y_train_p, dtype=tf.float32)
y_val_p_tf = tf.convert_to_tensor(y_val_p, dtype=tf.float32)
y_train_m_full_tf = tf.convert_to_tensor(y_train_m_full, dtype=tf.float32)
y_val_m_tf = tf.convert_to_tensor(y_val_m, dtype=tf.float32)

batch_size = 8

train_dataset_p = tf.data.Dataset.from_tensor_slices((dict(train_encodings_p), y_train_p_tf)).shuffle(1000).batch(batch_size)
val_dataset_p = tf.data.Dataset.from_tensor_slices((dict(val_encodings_p), y_val_p_tf)).batch(batch_size)

train_dataset_m_full = tf.data.Dataset.from_tensor_slices((dict(train_encodings_m_full), y_train_m_full_tf)).shuffle(1000).batch(batch_size)
val_dataset_m = tf.data.Dataset.from_tensor_slices((dict(val_encodings_m), y_val_m_tf)).batch(batch_size)

# -------------------------------
# 定义BERT模型
# -------------------------------
def create_bert_model():
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    return model

# 在政治数据上训练
model_p = create_bert_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model_p.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print("Training on Political News Dataset...")
history_p = model_p.fit(train_dataset_p, validation_data=val_dataset_p, epochs=2)

# 在迁移学习前对医疗数据集验证集进行评估
print("\nEvaluation on Medical News Validation Data before Transfer Learning:")
loss_m, accuracy_m = model_p.evaluate(val_dataset_m, verbose=0)
print(f"Medical News Validation Accuracy: {accuracy_m:.4f}")

print("Evaluation on Political News Validation Data before Transfer Learning:")
loss_p, accuracy_p = model_p.evaluate(val_dataset_p, verbose=0)
print(f"Political News Validation Accuracy: {accuracy_p:.4f}")

# 保存政治模型权重
model_p.save_weights("policy_news_bert_model.h5")

# 定义不同训练数据比例
fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
medical_accuracies = []

for frac in fractions:
    print(f"\nUsing {int(frac * 100)}% of the medical training data:")
    # 抽取部分医疗训练数据
    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(X_train_m_full, y_train_m_full, train_size=frac, random_state=42)
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # 编码
    train_encodings_m = encode_texts(X_train_m, tokenizer, max_length=maxlen)
    y_train_m_tf = tf.convert_to_tensor(y_train_m, dtype=tf.float32)
    train_dataset_m = tf.data.Dataset.from_tensor_slices((dict(train_encodings_m), y_train_m_tf)).shuffle(1000).batch(batch_size)

    # 创建新模型并加载权重
    model_m = create_bert_model()
    model_m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    model_m.load_weights("policy_news_bert_model.h5")

    # 在医疗数据上微调
    model_m.fit(train_dataset_m, validation_data=val_dataset_m, epochs=2, verbose=1)

    # 在医疗验证集上评估
    loss_m, accuracy_m = model_m.evaluate(val_dataset_m, verbose=0)
    medical_accuracies.append(accuracy_m)
    print(f"Medical News Validation Accuracy: {accuracy_m:.4f}")

    # 在政治验证集上评估
    loss_p, accuracy_p = model_m.evaluate(val_dataset_p, verbose=0)
    print(f"Political News Validation Accuracy: {accuracy_p:.4f}")

import matplotlib.pyplot as plt

# 绘制医疗验证集准确率与数据比例的关系
plt.figure()
plt.plot([f * 100 for f in fractions], medical_accuracies, marker='o')
plt.title('Medical News Validation Accuracy vs. Training Data Size (Transformer)')
plt.xlabel('Percentage of Medical Training Data Used (%)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()
