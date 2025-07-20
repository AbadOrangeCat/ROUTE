import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================================
# 1. Read Data
# ==========================================================
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

# Combine political texts + labels (0 = fake, 1 = real)
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([
    np.zeros(len(fake_texts)),
    np.ones(len(real_texts))
])

# Combine medical texts + labels (0 = fake, 1 = real)
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate([
    np.zeros(len(covid_fake_texts)),
    np.ones(len(covid_real_texts))
])

# ==========================================================
# 3. Fit Tokenizer ONLY on political news
#    (so the model initially knows nothing about medical data)
# ==========================================================
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(policy_texts)

# (Optional) save tokenizer for future inference
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# ==========================================================
# 4. Convert texts to sequences and pad them
# ==========================================================
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

maxlen = 1000
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

policy_labels = np.array(policy_labels)
medical_labels = np.array(medical_labels)

# ==========================================================
# 5. Split into training and validation sets
# ==========================================================
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
    policy_data, policy_labels, test_size=0.2, random_state=42
)

X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(
    medical_data, medical_labels, test_size=0.2, random_state=42
)


# ==========================================================
# 6. Build a minimal Transformer model
# ==========================================================

class TransformerBlock(layers.Layer):
    """
    A simple Transformer block using Multi-Head Attention + Feed Forward.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Multi-Head Self-Attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_model(
        vocab_size,
        embed_dim=64,
        maxlen=1000,
        num_heads=2,
        ff_dim=128,
        rate=0.1
):
    """
    Build a small Transformer-based model:
      - Embedding
      - Single TransformerBlock
      - Global Average Pooling
      - Dense layers
    """
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen)(inputs)

    # A single transformer block
    x = TransformerBlock(embed_dim, num_heads, ff_dim, rate=rate)(x)

    # Pooling + MLP
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ==========================================================
# 7. Utility functions for timing and evaluation
# ==========================================================
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


# ==========================================================
# 8. Pre-train on political news
# ==========================================================
print(">>> Start pre-training on political news...")
model_pre = create_transformer_model(
    vocab_size=len(tokenizer.word_index) + 1,
    embed_dim=64,
    maxlen=maxlen,
    num_heads=2,
    ff_dim=128,
    rate=0.1
)

history_pre, pretrain_time = timed_fit(
    model_pre, X_train_p, y_train_p,
    X_val_p, y_val_p,
    epochs=5,
    batch_size=64,
    verbose=1
)

print(f"Pre-training on political news took: {pretrain_time:.2f} seconds")

# Save pre-trained weights
model_pre.save_weights("policy_news_transformer.h5")

# Evaluate pre-trained model
print("\n=== Model performance before Transfer Learning ===")
print("Political News Validation:")
evaluate_and_print(model_pre, X_val_p, y_val_p, "Political Val")

print("Medical News Validation:")
evaluate_and_print(model_pre, X_val_m, y_val_m, "Medical Val")

# (Optional) Plot training curves
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_pre.history['accuracy'], label='Train Acc')
plt.plot(history_pre.history['val_accuracy'], label='Val Acc')
plt.title('Political Transformer Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_pre.history['loss'], label='Train Loss')
plt.plot(history_pre.history['val_loss'], label='Val Loss')
plt.title('Political Transformer Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================================
# 9. Compare Transfer Learning vs. Direct Training
#    for different fractions of medical data
# ==========================================================
fractions = [0.001, 0.002, 0.005,0.007,0.01]

transfer_accuracies = []
transfer_times = []
direct_accuracies = []
direct_times = []

for frac in fractions:
    print("\n" + "=" * 60)
    print(f"Using {int(frac * 1000)}% of the medical training data...")

    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(
            X_train_m_full, y_train_m_full,
            train_size=frac,
            random_state=42
        )
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # -------------------------
    # A) Transfer Learning
    # -------------------------
    print("[Transfer Learning] Fine-tuning the Transformer...")
    model_transfer = create_transformer_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embed_dim=64,
        maxlen=maxlen,
        num_heads=2,
        ff_dim=128,
        rate=0.1
    )
    # Load political pre-trained weights
    model_transfer.load_weights("policy_news_transformer.h5")

    _, ttime = timed_fit(
        model_transfer,
        X_train_m, y_train_m,
        X_val_m, y_val_m,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    transfer_times.append(ttime)
    print(f"[Transfer] Fine-tuning took: {ttime:.2f} seconds")

    print("[Transfer] Evaluating on medical validation set:")
    y_pred_prob = model_transfer.predict(X_val_m)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    acc_t = accuracy_score(y_val_m, y_pred)
    transfer_accuracies.append(acc_t)
    evaluate_and_print(model_transfer, X_val_m, y_val_m, "Medical Val (Transfer)")

    # -------------------------
    # B) Direct Training
    # -------------------------
    print("[Direct Training] Training from scratch (political + medical subset)...")
    X_train_direct = np.concatenate([X_train_p, X_train_m], axis=0)
    y_train_direct = np.concatenate([y_train_p, y_train_m], axis=0)

    model_direct = create_transformer_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embed_dim=64,
        maxlen=maxlen,
        num_heads=2,
        ff_dim=128,
        rate=0.1
    )

    _, dtime = timed_fit(
        model_direct,
        X_train_direct, y_train_direct,
        X_val_m, y_val_m,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    direct_times.append(dtime)
    print(f"[Direct] Training took: {dtime:.2f} seconds")

    print("[Direct] Evaluating on medical validation set:")
    y_pred_prob_direct = model_direct.predict(X_val_m)
    y_pred_direct = (y_pred_prob_direct > 0.5).astype("int32")
    acc_d = accuracy_score(y_val_m, y_pred_direct)
    direct_accuracies.append(acc_d)
    evaluate_and_print(model_direct, X_val_m, y_val_m, "Medical Val (Direct)")

# ==========================================================
# 10. Plot the comparison of Accuracy and Training Time
# ==========================================================
x_vals = [f * 100 for f in fractions]

plt.figure(figsize=(12, 5))

# 1) Accuracy
plt.subplot(1, 2, 1)
plt.plot(x_vals, transfer_accuracies, marker='o', label='Transfer Learning')
plt.plot(x_vals, direct_accuracies, marker='s', label='Direct Training')
plt.title('Medical Validation Accuracy (Transformer) vs. Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 2) Training Time
plt.subplot(1, 2, 2)
plt.plot(x_vals, transfer_times, marker='o', label='Transfer (fine-tuning)')
plt.plot(x_vals, direct_times, marker='s', label='Direct (from scratch)')
plt.title('Training Time (Transformer) vs. Medical Data Size')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print final summary
print("\n=== Final Summary ===")
for i, frac in enumerate(fractions):
    print(f"Medical fraction: {int(frac * 1000)}%")
    print(f" -> [Transfer] Acc={transfer_accuracies[i]:.4f}, Time={transfer_times[i]:.2f}s")
    print(f" -> [Direct]   Acc={direct_accuracies[i]:.4f}, Time={direct_times[i]:.2f}s")
    print("-" * 50)
