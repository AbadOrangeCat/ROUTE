import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# 1. 读取数据
# ======================
# Load fake and real news datasets
fake_df = pd.read_csv('../../news/fake.csv')  # Fake news dataset
real_df = pd.read_csv('../../news/true.csv')  # Real news dataset

# For medical news, we're reusing the same datasets as placeholders
covid_fake_df = pd.read_csv('../../news/fake.csv')  # Fake medical news dataset
covid_real_df = pd.read_csv('../../news/true.csv')  # Real medical news dataset

# Filter and extract political news texts
filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]
real_texts = filtered_df['text']

# Extract medical news texts
filtered_df = fake_df[(fake_df['subject'] == 'News') & (fake_df['text'].str.len() >= 40)]
covid_fake_texts = filtered_df['text']

filtered_df = real_df[(real_df['subject'] == 'worldnews') & (real_df['text'].str.len() >= 40)]
covid_real_texts = filtered_df['text']
# 政治文本+标签
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([
    np.zeros(len(fake_texts)),
    np.ones(len(real_texts))
])

# 医学文本+标签
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([
    np.zeros(len(covid_fake_texts)),
    np.ones(len(covid_real_texts))
])

# ======================
# 3. Tokenizer：只在政治新闻上fit
#    确保模型对医学新闻一无所知
# ======================
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(policy_texts)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# ======================
# 4. 文本->序列->padding
# ======================
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

maxlen = 1000
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

policy_labels = np.array(policy_labels)
medical_labels = np.array(medical_labels)

# ======================
# 5. 数据集拆分
# ======================
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
    policy_data, policy_labels, test_size=0.2, random_state=42
)
X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(
    medical_data, medical_labels, test_size=0.2, random_state=42
)

# ======================
# 6. 构建 LSTM 模型的函数
# ======================
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100


def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        input_length=maxlen))
    # 用LSTM替代Conv1D
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 评估函数
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


# 用于记录训练耗时
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
# 7. 先在政治新闻上做预训练
# ======================
print("开始在政治新闻上训练（预训练）...")
model_pre = create_model()
history_pre, pretrain_time = timed_fit(
    model_pre, X_train_p, y_train_p,
    X_val_p, y_val_p,
    epochs=5,  # 你可根据需要调整epoch
    batch_size=64,
    verbose=1
)
print(f"政治新闻预训练耗时: {pretrain_time:.2f} 秒")

# 保存预训练权重
model_pre.save_weights('policy_news_lstm_model.h5')

# 可选：观察训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_pre.history['accuracy'], label='Train Acc')
plt.plot(history_pre.history['val_accuracy'], label='Val Acc')
plt.title('Political (LSTM) Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_pre.history['loss'], label='Train Loss')
plt.plot(history_pre.history['val_loss'], label='Val Loss')
plt.title('Political (LSTM) Loss')
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
# 8. 不同医学数据量 -> 迁移学习 vs. 直接训练
#    + 比较训练时间
# =====================================================
fractions = [0.001, 0.002, 0.005,0.007,0.01]

medical_acc_transfer = []
medical_acc_direct = []
medical_time_transfer = []
medical_time_direct = []

for frac in fractions:
    print("\n" + "=" * 60)
    print(f"使用 {int(frac * 1000)}% 的医学训练数据...")

    # 采样医学训练数据
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
    # 加载政治新闻预训练权重
    model_transfer.load_weights('policy_news_lstm_model.h5')

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

    # 评估
    print("[迁移学习] 医学验证集结果:")
    y_pred_prob = model_transfer.predict(X_val_m)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    acc = accuracy_score(y_val_m, y_pred)
    medical_acc_transfer.append(acc)
    evaluate_and_print(model_transfer, X_val_m, y_val_m, "Medical Val (Transfer)")

    # -----------------------
    # B. 直接训练
    # -----------------------
    X_train_direct = np.concatenate([X_train_p, X_train_m], axis=0)
    y_train_direct = np.concatenate([y_train_p, y_train_m], axis=0)

    model_direct = create_model()
    print("[直接训练] 从零开始...")
    _, d_time = timed_fit(
        model_direct,
        X_train_direct, y_train_direct,
        X_val_m, y_val_m,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    medical_time_direct.append(d_time)
    print(f"[直接训练] 耗时: {d_time:.2f} 秒")

    # 评估
    print("[直接训练] 医学验证集结果:")
    y_pred_prob_direct = model_direct.predict(X_val_m)
    y_pred_direct = (y_pred_prob_direct > 0.5).astype("int32")
    acc_direct = accuracy_score(y_val_m, y_pred_direct)
    medical_acc_direct.append(acc_direct)
    evaluate_and_print(model_direct, X_val_m, y_val_m, "Medical Val (Direct)")

# ======================
# 9. 可视化对比
# ======================
x_vals = [f * 100 for f in fractions]

plt.figure(figsize=(12, 5))

# (1) Accuracy对比
plt.subplot(1, 2, 1)
plt.plot(x_vals, medical_acc_transfer, marker='o', label='Transfer Learning')
plt.plot(x_vals, medical_acc_direct, marker='s', label='Direct Training')
plt.title('Medical Val Accuracy (LSTM) vs. Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# (2) 训练时间对比
plt.subplot(1, 2, 2)
plt.plot(x_vals, medical_time_transfer, marker='o', label='Transfer (fine-tuning)')
plt.plot(x_vals, medical_time_direct, marker='s', label='Direct (from scratch)')
plt.title('Training Time (LSTM) vs. Medical Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== 最终结果汇总 ===")
for i, frac in enumerate(fractions):
    print(f"医学数据比例: {int(frac * 1000)}%")
    print(f" -> [迁移学习] Acc={medical_acc_transfer[i]:.4f}, Time={medical_time_transfer[i]:.2f}s")
    print(f" -> [直接训练] Acc={medical_acc_direct[i]:.4f}, Time={medical_time_direct[i]:.2f}s")
    print("-" * 50)
