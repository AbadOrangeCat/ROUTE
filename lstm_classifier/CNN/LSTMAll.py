import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入 TensorFlow 并检查 GPU 可用性
import tensorflow as tf
from tensorflow.keras import backend as K

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("TensorFlow will use the GPU for training.")
else:
    print("No GPU found. TensorFlow will use the CPU for training.")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split

# 读取假新闻和真新闻的数据集
fake_df = pd.read_csv('../news/fake.csv')  # 假新闻数据集
real_df = pd.read_csv('../news/true.csv')  # 真新闻数据集

covid_fake_df = pd.read_csv('../news/fake.csv')  # Fake medical news dataset
covid_real_df = pd.read_csv('../news/true.csv')  # Real medical news dataset


# 过滤并提取政治类的文本
filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]
real_texts = filtered_df['text']

# 提取医疗类的文本
filtered_df = fake_df[(fake_df['subject'] == 'News') & (fake_df['text'].str.len() >= 40)]
covid_fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'worldnews') & (real_df['text'].str.len() >= 40)]
covid_real_texts = filtered_df['text']

# 合并政治新闻文本和标签
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([np.zeros(len(fake_texts)), np.ones(len(real_texts))])  # 0表示假新闻，1表示真新闻

# 合并医疗新闻文本和标签
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([np.zeros(len(covid_fake_texts)), np.ones(len(covid_real_texts))])  # 0表示假新闻，1表示真新闻

# 初始化Tokenizer，并在所有文本上进行训练
all_texts = pd.concat([policy_texts, medical_texts])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(all_texts)

# 将文本转换为序列
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

# 填充序列
maxlen = 1000  # 序列的最大长度
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

# 标签数组化
policy_labels = np.array(policy_labels)
medical_labels = np.array(medical_labels)

# 将政治新闻数据集划分为训练集和验证集
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(policy_data, policy_labels, test_size=0.2, random_state=42)

# 将医疗新闻数据集划分为训练集和验证集
X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(medical_data, medical_labels, test_size=0.2,
                                                                    random_state=42)

# 构建 LSTM 模型
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小
embedding_dim = 100  # 词向量维度


def create_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 检查 GPU 可用性并设置设备
if tf.test.is_gpu_available():
    device_name = '/GPU:0'
else:
    device_name = '/CPU:0'

with tf.device(device_name):
    # 在政治新闻数据集上训练模型
    model = create_lstm_model()
    history = model.fit(X_train_p, y_train_p, epochs=5, batch_size=64, validation_data=(X_val_p, y_val_p))

# 绘制政治新闻模型的准确率和损失曲线
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('政治新闻模型准确率 (LSTM)')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper left')

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('政治新闻模型损失 (LSTM)')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(['训练集', '验证集'], loc='upper left')

plt.show()

# 保存模型
model.save('policy_news_lstm_model.h5')

# 在迁移学习前评估模型
print("迁移学习前在医疗新闻验证集上的评估：")
loss_m, accuracy_m = model.evaluate(X_val_m, y_val_m, verbose=0)
print(f"医疗新闻验证准确率: {accuracy_m:.4f}")

print("迁移学习前在政治新闻验证集上的评估：")
loss_p, accuracy_p = model.evaluate(X_val_p, y_val_p, verbose=0)
print(f"政治新闻验证准确率: {accuracy_p:.4f}")

# 定义不同的医疗数据集使用比例
fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # 1%, 5%, 10%, 20%, 50%, 100%
medical_accuracies = []
political_accuracies = []

for frac in fractions:
    print(f"\n使用 {int(frac * 100)}% 的医疗训练数据：")
    # 采样部分训练数据
    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(X_train_m_full, y_train_m_full, train_size=frac, random_state=42)
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # 创建新模型以确保公平比较
    model = create_lstm_model()

    # 加载在政治新闻上训练的模型权重
    model.load_weights('policy_news_lstm_model.h5')

    with tf.device(device_name):
        # 在采样的医疗新闻数据集上微调模型
        history_m = model.fit(X_train_m, y_train_m, epochs=5, batch_size=64, validation_data=(X_val_m, y_val_m),
                              verbose=0)

    # 在医疗新闻验证集上评估模型
    loss_m, accuracy_m = model.evaluate(X_val_m, y_val_m, verbose=0)
    medical_accuracies.append(accuracy_m)
    print(f"医疗新闻验证准确率: {accuracy_m:.4f}")

    # 在政治新闻验证集上评估模型
    loss_p, accuracy_p = model.evaluate(X_val_p, y_val_p, verbose=0)
    political_accuracies.append(accuracy_p)
    print(f"政治新闻验证准确率: {accuracy_p:.4f}")

# 绘制验证准确率与数据比例的关系图
plt.figure(figsize=(10, 5))
plt.plot([f * 100 for f in fractions], medical_accuracies, marker='o', label='医疗新闻')
plt.plot([f * 100 for f in fractions], political_accuracies, marker='x', label='政治新闻')
plt.title('验证准确率与医疗训练数据比例的关系 (LSTM)')
plt.xlabel('使用的医疗训练数据比例 (%)')
plt.ylabel('验证准确率')
plt.legend()
plt.grid(True)
plt.show()
