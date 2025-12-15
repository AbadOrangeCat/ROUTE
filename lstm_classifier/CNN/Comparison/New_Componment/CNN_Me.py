"""
Fair comparison of three training strategies on cross-domain fake news detection.

Curves:
1) Transfer (pretrain on political -> fine-tune on medical fraction). Time = pretrain + finetune.
2) Medical-from-scratch (from zero on the same medical fraction). Separate tokenizer.
3) Political+Medical-from-scratch (from zero on full political train + the same medical fraction). Separate tokenizer.

All three are evaluated on the SAME medical validation split (texts), but each method uses its own tokenizer,
so the val texts are vectorized under each method's tokenizer to avoid unfair vocabulary leakage.

Notes:
- Uses stratified splits everywhere for label balance.
- Tokenizers use OOV token and cap to MAX_NUM_WORDS.
- Seeds set for reproducibility.
- Uses only TensorFlow Keras (no mixed keras/tf.keras).
- Plots Accuracy & Total Time curves. Also prints Precision/Recall/F1.

Adjust DATA PATHS below to your files.
"""

import os
import random
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# 0. Reproducibility
# ======================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ======================
# 1. Config
# ======================
MAX_NUM_WORDS = 5000
MAXLEN = 1000
EMBED_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64

# Adjust paths here
PATH_FAKE = '../../../news/fake.csv'
PATH_REAL = '../../../news/true.csv'
PATH_COVID_FAKE = '../../../covid/fakeNews.csv'
PATH_COVID_REAL = '../../../covid/trueNews.csv'

# ======================
# 2. Utilities
# ======================

def build_tokenizer(train_texts: List[str], num_words: int = MAX_NUM_WORDS) -> Tuple[Tokenizer, int]:
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    vocab_size = min(num_words, len(tok.word_index) + 1)
    return tok, vocab_size


def vectorize(tok: Tokenizer, texts: List[str], maxlen: int = MAXLEN) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    data = pad_sequences(seqs, maxlen=maxlen)
    return data


def create_model(vocab_size: int, maxlen: int = MAXLEN,
                 embedding_dim: int = EMBED_DIM,
                 conv_filters: int = 128, kernel_size: int = 5,
                 dense_units: int = 10, dropout: float = 0.5) -> tf.keras.Model:
    m = Sequential()
    m.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    m.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu'))
    m.add(GlobalMaxPooling1D())
    m.add(Dense(units=dense_units, activation='relu'))
    m.add(Dropout(dropout))
    m.add(Dense(units=1, activation='sigmoid'))
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


def timed_fit(model: tf.keras.Model,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, verbose: int = 0) -> Tuple[tf.keras.callbacks.History, float]:
    start = time.time()
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_val, y_val), verbose=verbose)
    duration = time.time() - start
    return hist, duration


def evaluate(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray, name: str = "") -> dict:
    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype('int32')
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    print(f"{name} | Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def stratified_subset(texts: List[str], labels: np.ndarray, frac: float, seed: int = SEED) -> Tuple[List[str], np.ndarray]:
    if frac >= 1.0:
        return list(texts), np.array(labels)
    # train_test_split with stratify ensures class balance; handle edge cases gracefully
    try:
        X_sub, _, y_sub, _ = train_test_split(texts, labels, train_size=frac,
                                              random_state=seed, stratify=labels)
        return list(X_sub), np.array(y_sub)
    except ValueError:
        # Fallback: sample per-class with at least 1 per class
        texts = np.array(texts)
        idxs = np.arange(len(labels))
        chosen = []
        for cls in np.unique(labels):
            cls_idxs = idxs[labels == cls]
            n = max(1, int(np.floor(frac * len(cls_idxs))))
            n = min(n, len(cls_idxs))
            rng = np.random.RandomState(seed)
            chosen.extend(rng.choice(cls_idxs, size=n, replace=False))
        chosen = np.array(chosen)
        return texts[chosen].tolist(), labels[chosen]

# ======================
# 3. Load & filter data
# ======================
print("Loading datasets…")
fake_df = pd.read_csv(PATH_FAKE)
real_df = pd.read_csv(PATH_REAL)
covid_fake_df = pd.read_csv(PATH_COVID_FAKE)
covid_real_df = pd.read_csv(PATH_COVID_REAL)

# Political
pol_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
pol_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']

policy_texts = pd.concat([pol_fake, pol_real]).tolist()
policy_labels = np.concatenate([
    np.zeros(len(pol_fake), dtype=int),
    np.ones(len(pol_real), dtype=int)
])

print(f"Political | fake={len(pol_fake)}  real={len(pol_real)}  total={len(policy_texts)}")

# Medical (COVID) — assumes text column is 'Text'
med_fake = covid_fake_df[covid_fake_df['Text'].str.len() >= 40]['Text']
med_real = covid_real_df[covid_real_df['Text'].str.len() >= 40]['Text']

medical_texts = pd.concat([med_fake, med_real]).tolist()
medical_labels = np.concatenate([
    np.zeros(len(med_fake), dtype=int),
    np.ones(len(med_real), dtype=int)
])

print(f"Medical   | fake={len(med_fake)}   real={len(med_real)}   total={len(medical_texts)}")

# ======================
# 4. Train/Val Splits on TEXTS
# ======================
X_train_p_texts, X_val_p_texts, y_train_p, y_val_p = train_test_split(
    policy_texts, policy_labels, test_size=0.2, random_state=SEED, stratify=policy_labels
)
X_train_m_texts, X_val_m_texts, y_train_m_full, y_val_m = train_test_split(
    medical_texts, medical_labels, test_size=0.2, random_state=SEED, stratify=medical_labels
)

# ======================
# 5. Pretrain on POLITICAL (for Transfer curve)
# ======================
print("\nPretraining on political news…")
TOK_P, VOCAB_P = build_tokenizer(X_train_p_texts, MAX_NUM_WORDS)
X_train_p = vectorize(TOK_P, X_train_p_texts, MAXLEN)
X_val_p = vectorize(TOK_P, X_val_p_texts, MAXLEN)

model_pre = create_model(VOCAB_P, MAXLEN, EMBED_DIM)
hist_pre, pretrain_time = timed_fit(model_pre, X_train_p, y_train_p, X_val_p, y_val_p,
                                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
print(f"Pretrain time: {pretrain_time:.2f}s")
pretrained_weights = model_pre.get_weights()

# Optional: zero-shot check on medical val
X_val_m_for_pre = vectorize(TOK_P, X_val_m_texts, MAXLEN)
_ = evaluate(model_pre, X_val_m_for_pre, y_val_m, name="Zero-shot (political->medical)")

# ======================
# 6. Fractions loop
# ======================
fractions = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

acc_transfer, f1_transfer, time_transfer_total = [], [], []
acc_med_scratch, f1_med_scratch, time_med_scratch = [], [], []
acc_combined_scratch, f1_combined_scratch, time_combined_scratch = [], [], []

for frac in fractions:
    print("\n" + "="*68)
    print(f"Using {int(frac*100)}% of MEDICAL train data…")

    # Sample medical subset (texts)
    X_train_m_sub_texts, y_train_m_sub = stratified_subset(X_train_m_texts, y_train_m_full, frac, seed=SEED)

    # --------------------
    # (1) Transfer: political pretrain -> fine-tune on MED subset
    # --------------------
    print("[Transfer] Fine-tuning…")
    X_train_m_sub = vectorize(TOK_P, X_train_m_sub_texts, MAXLEN)
    X_val_m_for_transfer = vectorize(TOK_P, X_val_m_texts, MAXLEN)

    model_transfer = create_model(VOCAB_P, MAXLEN, EMBED_DIM)
    model_transfer.set_weights(pretrained_weights)
    _, t_fine = timed_fit(model_transfer, X_train_m_sub, y_train_m_sub,
                          X_val_m_for_transfer, y_val_m,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    time_transfer_total.append(pretrain_time + t_fine)
    met = evaluate(model_transfer, X_val_m_for_transfer, y_val_m, name="Medical Val (Transfer)")
    acc_transfer.append(met['acc'])
    f1_transfer.append(met['f1'])

    # --------------------
    # (2) Medical-from-scratch: tokenizer & model from zero on MED subset
    # --------------------
    print("[Med-Scratch] Training from scratch…")
    TOK_M, VOCAB_M = build_tokenizer(X_train_m_sub_texts, MAX_NUM_WORDS)
    X_train_m_sub_scratch = vectorize(TOK_M, X_train_m_sub_texts, MAXLEN)
    X_val_m_for_med_scratch = vectorize(TOK_M, X_val_m_texts, MAXLEN)

    model_med = create_model(VOCAB_M, MAXLEN, EMBED_DIM)
    _, t_med = timed_fit(model_med, X_train_m_sub_scratch, y_train_m_sub,
                         X_val_m_for_med_scratch, y_val_m,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    time_med_scratch.append(t_med)
    met = evaluate(model_med, X_val_m_for_med_scratch, y_val_m, name="Medical Val (Med-Scratch)")
    acc_med_scratch.append(met['acc'])
    f1_med_scratch.append(met['f1'])

    # --------------------
    # (3) Political+Medical-from-scratch: tokenizer & model from zero on POL train + MED subset
    # --------------------
    print("[Pol+Med-Scratch] Training from scratch…")
    comb_train_texts = list(X_train_p_texts) + list(X_train_m_sub_texts)
    comb_train_labels = np.concatenate([y_train_p, y_train_m_sub])

    TOK_PM, VOCAB_PM = build_tokenizer(comb_train_texts, MAX_NUM_WORDS)
    X_train_comb = vectorize(TOK_PM, comb_train_texts, MAXLEN)
    X_val_m_for_combined = vectorize(TOK_PM, X_val_m_texts, MAXLEN)

    model_comb = create_model(VOCAB_PM, MAXLEN, EMBED_DIM)
    _, t_comb = timed_fit(model_comb, X_train_comb, comb_train_labels,
                          X_val_m_for_combined, y_val_m,
                          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    time_combined_scratch.append(t_comb)
    met = evaluate(model_comb, X_val_m_for_combined, y_val_m, name="Medical Val (Pol+Med-Scratch)")
    acc_combined_scratch.append(met['acc'])
    f1_combined_scratch.append(met['f1'])

# ======================
# 7. Plotting
# ======================
x_vals = [f*100 for f in fractions]
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(x_vals, acc_transfer, marker='o', label='Transfer (pretrain+finetune)')
plt.plot(x_vals, acc_med_scratch, marker='s', label='Med from scratch')
plt.plot(x_vals, acc_combined_scratch, marker='^', label='Pol+Med from scratch')
plt.title('Medical Validation Accuracy vs. Training Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Time (seconds)
plt.subplot(1,2,2)
plt.plot(x_vals, time_transfer_total, marker='o', label='Transfer (total = pretrain + finetune)')
plt.plot(x_vals, time_med_scratch, marker='s', label='Med from scratch')
plt.plot(x_vals, time_combined_scratch, marker='^', label='Pol+Med from scratch')
plt.title('Training Time vs. Medical Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================
# 8. Summary table
# ======================
print("\n=== Summary (Medical Validation) ===")
for i, frac in enumerate(fractions):
    print(f"Medical data fraction: {int(frac*100)}%")
    print(f" -> Transfer           Acc={acc_transfer[i]:.4f}  F1={f1_transfer[i]:.4f}  Time(total)={time_transfer_total[i]:.2f}s")
    print(f" -> Med from scratch   Acc={acc_med_scratch[i]:.4f}  F1={f1_med_scratch[i]:.4f}  Time={time_med_scratch[i]:.2f}s")
    print(f" -> Pol+Med from scratch Acc={acc_combined_scratch[i]:.4f}  F1={f1_combined_scratch[i]:.4f}  Time={time_combined_scratch[i]:.2f}s")
    print("-"*60)

print("\nTip: If you want to exclude pretrain time from the Transfer curve (e.g., amortized across many targets),\njust plot 't_fine' instead of 'pretrain_time + t_fine'.")
