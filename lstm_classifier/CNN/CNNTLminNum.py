import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# 1. 读取数据
# ======================
fake_df = pd.read_csv('../news/fake.csv')  # 假新闻数据集
real_df = pd.read_csv('../news/true.csv')  # 真新闻数据集

# 这里使用相同数据集来模拟医学新闻
covid_fake_df = pd.read_csv('../news/fake.csv')  # 假医学新闻
covid_real_df = pd.read_csv('../news/true.csv')  # 真医学新闻

# ======================
# 2. 筛选并分别取出 政治类 和 医学类 文本
# ======================

# 政治新闻：假新闻部分
filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

# 政治新闻：真新闻部分
filtered_df = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]
real_texts = filtered_df['text']

# 医学新闻：假新闻部分
filtered_df = fake_df[(fake_df['subject'] == 'News') & (fake_df['text'].str.len() >= 40)]
covid_fake_texts = filtered_df['text']

# 医学新闻：真新闻部分
filtered_df = real_df[(real_df['subject'] == 'worldnews') & (real_df['text'].str.len() >= 40)]
covid_real_texts = filtered_df['text']

# 组合政治新闻文本 + 标签（0=假，1=真）
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([
    np.zeros(len(fake_texts)),
    np.ones(len(real_texts))
])

# 组合医学新闻文本 + 标签（0=假，1=真）
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([
    np.zeros(len(covid_fake_texts)),
    np.ones(len(covid_real_texts))
])

# ======================
# 3. 只在一开始对所有文本 fit Tokenizer
# ======================
all_texts = pd.concat([policy_texts, medical_texts])

tokenizer = Tokenizer(num_words=5000)  # 使用前5000词（可根据需要调整）
tokenizer.fit_on_texts(all_texts)

# （可选）保存 tokenizer，用于后续单独推理时使用
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# ======================
# 4. 将文本转换为序列并做 padding
# ======================
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

maxlen = 1000  # 每个样本统一到此长度
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

# 标签转为数组
policy_labels = np.array(policy_labels)
medical_labels = np.array(medical_labels)

# ======================
# 5. 拆分训练集和验证集
# ======================
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
    policy_data, policy_labels, test_size=0.2, random_state=42
)

X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(
    medical_data, medical_labels, test_size=0.2, random_state=42
)

# ======================
# 6. 构建模型的函数
# ======================
vocab_size = len(tokenizer.word_index) + 1  # 注意这里使用fit之后的tokenizer，保证与后续相同
embedding_dim = 100  # 词向量维度


def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=maxlen))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======================
# 7. 先在政治新闻数据上训练
# ======================
model = create_model()
history = model.fit(
    X_train_p, y_train_p,
    epochs=5,
    batch_size=64,
    validation_data=(X_val_p, y_val_p)
)

# 保存在政治新闻上训练的权重
model.save_weights('policy_news_cnn_model.h5')

# 画图：可视化准确率和损失
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Political News Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Political News Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# ======================
# 8. 在做迁移学习前, 在政治/医学验证集上评估(含准确率等4个指标)
# ======================
def evaluate_and_print(model, X_val, y_val, name=""):
    """对模型在指定验证集X_val, y_val上进行预测，并打印4个指标。"""
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"{name} Accuracy:  {acc:.4f}")
    print(f"{name} Precision: {prec:.4f}")
    print(f"{name} Recall:    {rec:.4f}")
    print(f"{name} F1-score:  {f1:.4f}")
    print("-" * 40)


print("=== Before Transfer Learning ===")
print("Political News Validation:")
evaluate_and_print(model, X_val_p, y_val_p, "Political Val")

print("Medical News Validation:")
evaluate_and_print(model, X_val_m, y_val_m, "Medical Val")

# ======================
# 9. 不再重新 fit tokenizer。
#    下面进行不同数据量的医学新闻训练(迁移学习)
# ======================
fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
medical_accuracies = []

print("=== Transfer Learning (Medical) ===")
for frac in fractions:
    print(f"\nUsing {int(frac * 100)}% of the medical training data:")

    # 按照 frac 采样一部分医学训练集
    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(
            X_train_m_full, y_train_m_full,
            train_size=frac,
            random_state=42
        )
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # 每次新建同样结构的模型
    model = create_model()
    # 加载政治新闻训练的初始权重
    model.load_weights('policy_news_cnn_model.h5')

    # 在医学新闻训练集中做微调
    model.fit(
        X_train_m, y_train_m,
        epochs=5,
        batch_size=64,
        validation_data=(X_val_m, y_val_m),
        verbose=0
    )

    # 评估在医学新闻上的指标
    print("Medical News Validation after fine-tuning:")
    evaluate_and_print(model, X_val_m, y_val_m, "Medical Val")

    # 我们记录 Accuracy 以便后面绘图
    y_pred_m = (model.predict(X_val_m) > 0.5).astype("int32")
    acc_m = accuracy_score(y_val_m, y_pred_m)
    medical_accuracies.append(acc_m)

    # 评估在政治新闻验证集上的指标(看看微调后对原任务的影响)
    print("Political News Validation after fine-tuning:")
    evaluate_and_print(model, X_val_p, y_val_p, "Political Val")

# ======================
# 10. 可视化 不同医学训练数据量 对医学验证集Accuracy的影响
# ======================
plt.figure()
plt.plot([f * 100 for f in fractions], medical_accuracies, marker='o')
plt.title('Medical News Validation Accuracy vs. Training Data Size')
plt.xlabel('Percentage of Medical Training Data Used (%)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()
