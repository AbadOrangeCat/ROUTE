import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# 1. 读取数据
# ======================
fake_df = pd.read_csv('../../news/fake.csv')  # Fake news dataset
real_df = pd.read_csv('../../news/true.csv')  # Real news dataset

# (Using the same dataset to simulate medical news)
covid_fake_df = pd.read_csv('../../covid/fakeNews.csv')  # Fake medical news
covid_real_df = pd.read_csv('../../covid/trueNews.csv')  # Real medical news

# ==========================================================
# 2. Filter out texts for Political and Medical categories
# ==========================================================
# Political: fake
filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

# Political: real
filtered_df = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]
real_texts = filtered_df['text']

# Medical: fake
filtered_df = covid_fake_df[(covid_fake_df['Text'].str.len() >= 40)]
covid_fake_texts = filtered_df['Text']

# Medical: real
filtered_df = covid_real_df[(covid_real_df['Text'].str.len() >= 40)]
covid_real_texts = filtered_df['Text']


# 组合政治新闻文本 + 标签（0=假，1=真）
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([
    np.zeros(len(fake_texts)),
    np.ones(len(real_texts))
])

print("假文本数量:", len(fake_texts))
print("真文本数量:", len(real_texts))
print("总数量:", len(policy_texts))


# 组合医学新闻文本 + 标签（0=假，1=真）
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([
    np.zeros(len(covid_fake_texts)),
    np.ones(len(covid_real_texts))
])

print("假文本数量:", len(covid_fake_texts))
print("真文本数量:", len(covid_real_texts))
print("总数量:", len(medical_texts))


# ======================
# 3. 只用 政治新闻文本 来 fit Tokenizer
#    保证模型在最初对医学新闻一无所知
# ======================
tokenizer = Tokenizer(num_words=5000)  # 使用前5000词（可根据需要调整）
tokenizer.fit_on_texts(policy_texts)

# （可选）保存 tokenizer，用于后续单独推理时使用
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# ======================
# 4. 将政治/医学文本转换为序列并做 padding
# ======================
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

maxlen = 1000  # 每个样本统一到此长度
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

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
vocab_size = len(tokenizer.word_index) + 1
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

def evaluate_and_print(model, X_val, y_val, name=""):
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

# 简单的工具函数，记录训练耗时
def timed_fit(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64, verbose=1):
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=verbose
    )
    end_time = time.time()
    training_time = end_time - start_time
    return history, training_time


# ======================
# 7. 在政治新闻数据上训练 (预训练)
# ======================
model_pre = create_model()
print("开始在政治新闻上训练（预训练）...")
history_pre, pretrain_time = timed_fit(
    model_pre, X_train_p, y_train_p,
    X_val_p, y_val_p,
    epochs=5,
    batch_size=64,
    verbose=1
)
print(f"政治新闻预训练耗时: {pretrain_time:.2f} 秒")

# 保存在政治新闻上训练的权重
model_pre.save_weights('policy_news_cnn_model.h5')

# 可选：画图观察训练过程
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_pre.history['accuracy'], label='Train Acc')
plt.plot(history_pre.history['val_accuracy'], label='Val Acc')
plt.title('Political News Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_pre.history['loss'], label='Train Loss')
plt.plot(history_pre.history['val_loss'], label='Val Loss')
plt.title('Political News Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

print("=== Model performance before Transfer Learning ===")
print("Political News Validation:")
evaluate_and_print(model_pre, X_val_p, y_val_p, "Political Val")

print("Medical News Validation:")
evaluate_and_print(model_pre, X_val_m, y_val_m, "Medical Val")

# =====================================================
# 8. 不同数据量的医学新闻 -> 迁移学习 vs. 直接训练 + 训练时间比较
# =====================================================
fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
# fractions = [0.0001, 0.0002, 0.0005,0.0007,0.001]
medical_acc_transfer = []
medical_acc_direct = []
medical_time_transfer = []
medical_time_direct = []

for frac in fractions:
    print("\n" + "="*60)
    print(f"使用 {int(frac*100)}% 的医学训练数据...")

    # 挑选 frac 比例的医学训练数据
    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(
            X_train_m_full, y_train_m_full,
            train_size=frac,
            random_state=42
        )
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # -----------------------
    # A. 迁移学习
    # -----------------------
    model_transfer = create_model()
    # 加载政治新闻预训练的初始权重
    model_transfer.load_weights('policy_news_cnn_model.h5')

    # 微调
    print("[迁移学习] 开始微调...")
    _, t_time = timed_fit(
        model_transfer,
        X_train_m, y_train_m,
        X_val_m, y_val_m,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    medical_time_transfer.append(t_time)
    print(f"[迁移学习] 微调耗时: {t_time:.2f} 秒")

    # 评估迁移学习模型
    print("[迁移学习] 在医学验证集上的结果:")
    y_pred_prob = model_transfer.predict(X_val_m)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    acc_m = accuracy_score(y_val_m, y_pred)
    medical_acc_transfer.append(acc_m)
    evaluate_and_print(model_transfer, X_val_m, y_val_m, "Medical Val (Transfer)")

    # -----------------------
    # B. 直接训练
    # -----------------------
    # 将全部政治训练数据 + 采样后的医学训练数据合并
    X_train_direct = np.concatenate([X_train_p, X_train_m], axis=0)
    y_train_direct = np.concatenate([y_train_p, y_train_m], axis=0)
    #
    # model_direct = create_model()
    # print("[直接训练] 开始从零训练...")
    # _, d_time = timed_fit(
    #     model_direct,
    #     X_train_direct, y_train_direct,
    #     X_val_m, y_val_m,
    #     epochs=5,
    #     batch_size=64,
    #     verbose=0
    # )
    # medical_time_direct.append(d_time)
    # print(f"[直接训练] 耗时: {d_time:.2f} 秒")
    #
    # print("[直接训练] 在医学验证集上的结果:")
    # y_pred_prob_direct = model_direct.predict(X_val_m)
    # y_pred_direct = (y_pred_prob_direct > 0.5).astype("int32")
    # acc_m_direct = accuracy_score(y_val_m, y_pred_direct)
    # medical_acc_direct.append(acc_m_direct)
    # evaluate_and_print(model_direct, X_val_m, y_val_m, "Medical Val (Direct)")

# ======================
# 9. 可视化对比
# ======================
x_vals = [f*100 for f in fractions]

plt.figure(figsize=(12,5))

# (1) Accuracy对比
plt.subplot(1,2,1)
plt.plot(x_vals, medical_acc_transfer, marker='o', label='Transfer Learning')
plt.plot(x_vals, medical_acc_direct, marker='s', label='Direct Training')
plt.title('Medical Validation Accuracy vs. Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# (2) 训练时间对比
plt.subplot(1,2,2)
plt.plot(x_vals, medical_time_transfer, marker='o', label='Transfer (fine-tuning)')
plt.plot(x_vals, medical_time_direct, marker='s', label='Direct (from scratch)')
plt.title('Training Time vs. Medical Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印最终结果总结
print("\n=== 最终结果汇总 ===")
for i, frac in enumerate(fractions):
    print(f"医学数据比例: {int(frac*100)}%")
    print(f" -> [迁移学习] Accuracy={medical_acc_transfer[i]:.4f}, Time={medical_time_transfer[i]:.2f}s")
    print(f" -> [直接训练] Accuracy={medical_acc_direct[i]:.4f}, Time={medical_time_direct[i]:.2f}s")
    print("-"*50)
