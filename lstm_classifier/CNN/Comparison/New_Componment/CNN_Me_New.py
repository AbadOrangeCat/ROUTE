"""
Fair comparison + Domain-Aware Adapters + EWC (Elastic Weight Consolidation)
--------------------------------------------------------------------------
This script extends the previous three-curve benchmark with two innovations:
  1) Domain-aware lightweight adapters (gated by a domain embedding) to explicitly model domain differences.
  2) Source-knowledge preserving regularization with EWC during fine-tuning to reduce catastrophic forgetting.

Curves:
1) Transfer+Adapter+EWC : Pretrain on political (domain=0), then fine-tune on medical fraction (domain=1),
                          with domain-aware adapters enabled and EWC penalty toward source model.
2) Med-from-scratch (Plain): Train from zero on the same medical fraction using a plain TextCNN (no adapters, no EWC).
3) Pol+Med-from-scratch (Plain): Train from zero on political train + medical fraction using a plain TextCNN (no adapters, no EWC).
3) Pol+Med-from-scratch+A: Train from zero on political train + medical fraction, SAME architecture, WITHOUT EWC.

Notes
- Uses a single tokenizer fit on POLITICAL TRAIN TEXTS (same as original baseline) to keep the embedding size fixed
  and make EWC easier to apply to the whole network. If you want union tokenizers per fraction, see TODO at bottom.
- Domain-aware adapters are tiny bottleneck modules with residual connections, plus a domain-conditioned channel gate.
- EWC penalty is applied to ALL non-Embedding variables by name match (we skip the Embedding matrix for simplicity).
- Metric threshold uses 0.5 by default; optional F1-based threshold calibration is provided.

Adjust DATA PATHS below to your files.
"""

import os
import random
import time
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers as L
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

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

# Domain-aware Adapter config
USE_DOMAIN_ADAPTER = True
ADAPTER_R = 16               # bottleneck size
GATE_HIDDEN = 64             # hidden size for domain gate

# EWC config (used ONLY for Transfer curve)
USE_EWC = True
EWC_LAMBDA = 23       # strength of EWC penalty (tune: 10~200)
FISHER_SAMPLES = 2000       # political samples to estimate Fisher (will be clipped to available)
FISHER_BATCH_SIZE = 64

# Threshold calibration
CALIBRATE_THRESHOLD = True  # if True, pick threshold that maximizes F1 on the validation set for reporting

# Adjust data paths here
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


def domain_ids_like(length: int, domain:int) -> np.ndarray:
    return np.full((length,), domain, dtype=np.int32)


def evaluate_probs(y_prob: np.ndarray, y_true: np.ndarray, name: str = "", calibrate: bool = CALIBRATE_THRESHOLD) -> Dict[str, float]:
    if calibrate:
        P, R, T = precision_recall_curve(y_true, y_prob)
        F1 = 2*P*R/(P+R+1e-9)
        best_idx = np.argmax(F1[:-1])
        thr = T[best_idx]
    else:
        thr = 0.5
    y_pred = (y_prob > thr).astype('int32')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name} | thr={thr:.3f}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "thr": thr}


def stratified_subset_list(texts: List[str], labels: np.ndarray, frac: float, seed: int = SEED) -> Tuple[List[str], np.ndarray]:
    """Return a stratified subset of (texts, labels) of size `frac`.
    If frac >= 1.0, return the full set. Robust to sklearn's train_size restriction (must be <1.0).
    """
    if frac >= 1.0:
        return list(texts), np.array(labels)
    if frac <= 0.0:
        raise ValueError("frac must be > 0.0")
    X_sub, _, y_sub, _ = train_test_split(texts, labels, train_size=frac, random_state=seed, stratify=labels)
    return list(X_sub), np.array(y_sub)


# ======================
# 3. Domain-aware Adapter CNN
# ======================

def build_domain_adapter_cnn(vocab_size: int,
                             maxlen: int = MAXLEN,
                             embed_dim: int = EMBED_DIM,
                             conv_filters: int = 128,
                             kernel_size: int = 5,
                             dense_units: int = 10,
                             dropout: float = 0.5,
                             use_adapter: bool = USE_DOMAIN_ADAPTER,
                             adapter_r: int = ADAPTER_R,
                             gate_hidden: int = GATE_HIDDEN) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    dom_in = L.Input(shape=(), dtype='int32', name='domain_id')  # 0=political, 1=medical

    x = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, name='emb')(tok_in)

    h = L.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', name='conv')(x)

    if use_adapter:
        # Houlsby-style bottleneck adapter (time-distributed), residual
        a = L.TimeDistributed(L.Dense(adapter_r, activation='relu'), name='adapter_down')(h)
        a = L.TimeDistributed(L.Dense(conv_filters), name='adapter_up')(a)
        h = L.Add(name='adapter_residual')([h, a])

        # Domain-conditioned channel gate (SE-style), residual scaling
        # domain embedding → small MLP → sigmoid gate over channels
        d = L.Embedding(input_dim=2, output_dim=gate_hidden, name='dom_emb')(dom_in)
        d = L.Dense(gate_hidden, activation='relu', name='dom_proj1')(d)
        d = L.Dense(conv_filters, activation=None, name='dom_proj2')(d)  # (B, C)
        d = L.Reshape((1, conv_filters))(d)  # broadcast across time

        g_stat = L.GlobalAveragePooling1D(name='gap')(h)                  # (B, C)
        g_stat = L.Dense(conv_filters, activation=None, name='stat_proj')(g_stat)
        g_stat = L.Reshape((1, conv_filters))(g_stat)

        gate_preact = L.Add(name='gate_preact')([d, g_stat])
        gate = L.Activation('sigmoid', name='gate_sigmoid')(gate_preact)
        h = L.Multiply(name='gated')([h, gate])

    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(units=dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(units=1, activation='sigmoid', name='out')(z)

    model = Model(inputs={'tokens': tok_in, 'domain_id': dom_in}, outputs=out, name='DomainAdapterCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_plain_cnn(vocab_size: int,
                     maxlen: int = MAXLEN,
                     embed_dim: int = EMBED_DIM,
                     conv_filters: int = 128,
                     kernel_size: int = 5,
                     dense_units: int = 10,
                     dropout: float = 0.5) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    x = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', name='conv')(x)
    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(units=dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(units=1, activation='sigmoid', name='out')(z)
    model = Model(inputs=tok_in, outputs=out, name='PlainCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================
# 4. EWC wrapper model
# ======================
class EWCModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model,
                 fisher: Dict[str, tf.Tensor],
                 theta_old: Dict[str, tf.Tensor],
                 ewc_lambda: float = EWC_LAMBDA,
                 exclude_patterns: Tuple[str, ...] = ("emb",),
                 alpha=0.0):
        """Wrap a base model and add an EWC penalty during training.
        exclude_patterns: variable name substrings to skip (e.g., embedding matrix).
        """
        super().__init__()
        self.base = base_model
        self.fisher = fisher
        self.theta_old = theta_old
        self.ewc_lambda = ewc_lambda
        self.exclude = exclude_patterns
        self.alpha = alpha


    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32); y = tf.reshape(y, (-1,1))
        with tf.GradientTape() as tape:
            y_pred = self.base(x, training=True)
            loss = self.compiled_loss(y, y_pred)

            ewc_pen, prox_pen = 0.0, 0.0
            for v in self.base.trainable_variables:
                name = v.name
                if name in self.fisher and name in self.theta_old:
                    delta = v - self.theta_old[name]
                    ewc_pen += tf.reduce_sum(self.fisher[name] * tf.square(delta))
                    prox_pen += tf.reduce_sum(tf.square(delta))  # 可选 α 项
            loss += 0.5 * self.ewc_lambda * ewc_pen + 0.5 * self.alpha * prox_pen

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))
        y_pred = self.base(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

EWC_EXCLUDE_PATTERNS = ("emb", "adapter_", "dom_", "gate_", "stat_proj")

def compute_fisher_diagonal(model, X_tokens, y, domain_ids, max_samples=FISHER_SAMPLES, batch_size=FISHER_BATCH_SIZE):
    n = min(len(y), max_samples)
    idx = np.random.choice(len(y), size=n, replace=False)
    Xt, yt, dt = X_tokens[idx], y[idx].astype(np.float32).reshape(-1,1), domain_ids[idx].astype(np.int32)

    # 只对共享主干变量建 Fisher
    shared_vars = [v for v in model.trainable_variables
                   if not any(ex in v.name for ex in EWC_EXCLUDE_PATTERNS)]
    fisher = {v.name: tf.zeros_like(v) for v in shared_vars}

    ds = tf.data.Dataset.from_tensor_slices(({"tokens": Xt, "domain_id": dt}, yt)).batch(batch_size)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    for x_b, y_b in ds:
        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=False)
            nll = tf.reduce_mean(bce(tf.cast(y_b, y_pred.dtype), y_pred))
        grads = tape.gradient(nll, shared_vars)
        for v, g in zip(shared_vars, grads):
            if g is not None:
                fisher[v.name] = fisher[v.name] + tf.square(g)

    num_batches = tf.cast(tf.math.ceil(n / batch_size), fisher[next(iter(fisher))].dtype)
    for name in fisher:
        fisher[name] = fisher[name] / num_batches
    return fisher


def snapshot_weights(model: tf.keras.Model) -> Dict[str, tf.Tensor]:
    return {v.name: tf.identity(v) for v in model.trainable_variables
            if not any(ex in v.name for ex in EWC_EXCLUDE_PATTERNS)}

# ======================
# 5. Load & filter data
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
# 6. Train/Val splits on TEXTS (stratified)
# ======================
X_train_p_texts, X_val_p_texts, y_train_p, y_val_p = train_test_split(
    policy_texts, policy_labels, test_size=0.2, random_state=SEED, stratify=policy_labels
)
X_train_m_texts, X_val_m_texts, y_train_m_full, y_val_m = train_test_split(
    medical_texts, medical_labels, test_size=0.2, random_state=SEED, stratify=medical_labels
)

# ======================
# 7. Tokenizer (fit on POLITICAL TRAIN only)
# ======================
TOK_P, VOCAB_P = build_tokenizer(X_train_p_texts, MAX_NUM_WORDS)

# Vectorize political splits with TOK_P
X_train_p = vectorize(TOK_P, X_train_p_texts, MAXLEN)
X_val_p = vectorize(TOK_P, X_val_p_texts, MAXLEN)

# For medical val, we will also use TOK_P to keep embedding size fixed (see Notes)
X_val_m_tokP = vectorize(TOK_P, X_val_m_texts, MAXLEN)

# Domain id arrays
D_train_p = domain_ids_like(len(X_train_p), 0)
D_val_p = domain_ids_like(len(X_val_p), 0)
D_val_m = domain_ids_like(len(X_val_m_tokP), 1)

# ======================
# 8. Pretrain on POLITICAL with Domain Adapters (domain=0)
# ======================
print("Pretraining on political news with domain-aware adapters…")
pre_model = build_domain_adapter_cnn(VOCAB_P)
pre_model.summary()

start = time.time()
pre_model.fit({"tokens": X_train_p, "domain_id": D_train_p}, y_train_p,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=({"tokens": X_val_p, "domain_id": D_val_p}, y_val_p),
              verbose=1)
pretrain_time = time.time() - start
print(f"Pretrain time: {pretrain_time:.2f}s")

# Zero-shot on medical val
zp = pre_model.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
_ = evaluate_probs(zp, y_val_m, name="Zero-shot (political->medical)")

# EWC stats (Fisher & snapshot) on political
if USE_EWC:
    print("Computing Fisher information on source (political)…")
    fisher = compute_fisher_diagonal(pre_model, X_train_p, y_train_p, D_train_p,
                                     max_samples=FISHER_SAMPLES, batch_size=FISHER_BATCH_SIZE)
    theta_old = snapshot_weights(pre_model)
else:
    fisher, theta_old = {}, {}

# Keep pretrained weights for reuse
pre_weights = pre_model.get_weights()

# ======================
# 9. Fractions loop
# ======================
fractions = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

acc_transfer, f1_transfer, time_transfer_total,fine_turn = [], [], [],[]
acc_med_scratch, f1_med_scratch, time_med_scratch = [], [], []
acc_combined_scratch, f1_combined_scratch, time_combined_scratch = [], [], []
for frac in fractions:
    print("\n" + "="*68)
    print(f"Using {int(frac*100)}% of MEDICAL train data…")

    # (a) subset medical train texts
    X_train_m_texts_sub, y_train_m_sub = stratified_subset_list(
        X_train_m_texts, y_train_m_full, frac, SEED
    )

    # Vectorize medical subset using TOK_P (fixed vocab)
    X_train_m_sub_tokP = vectorize(TOK_P, X_train_m_texts_sub, MAXLEN)
    D_train_m = domain_ids_like(len(X_train_m_sub_tokP), 1)

    # --------------------
    # (1) Transfer + Adapter + (optional) EWC
    # --------------------
    print("[Transfer+Adapter+EWC] Fine-tuning...")
    transfer_model = build_domain_adapter_cnn(VOCAB_P)
    transfer_model.set_weights(pre_weights)

    if USE_EWC:
        ewc_model = EWCModel(
            transfer_model, fisher=fisher, theta_old=theta_old, ewc_lambda=EWC_LAMBDA
        )
        ewc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        ewc_model.fit(
            {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=({"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m),
            verbose=0
        )
        t_fine = time.time() - start
        y_prob = ewc_model.predict(
            {"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0
        ).ravel()
    else:
        start = time.time()
        transfer_model.fit(
            {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=({"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m),
            verbose=0
        )
        t_fine = time.time() - start
        y_prob = transfer_model.predict(
            {"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0
        ).ravel()

    time_transfer_total.append(pretrain_time)   # 你现在是把预训练与微调分开画
    fine_turn.append(t_fine)                    # 只画微调耗时
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val (Transfer+Adapter+EWC)")
    acc_transfer.append(met['acc'])
    f1_transfer.append(met['f1'])

    # --------------------
    # (2) Pol+Med-from-scratch (Plain)
    # --------------------
    print("[Pol+Med-Scratch (Plain)] Training from scratch...")
    # Build concatenated train set (political + medical subset)
    X_train_comb = np.concatenate([X_train_p, X_train_m_sub_tokP], axis=0)
    y_train_comb = np.concatenate([y_train_p, y_train_m_sub], axis=0)

    comb_model = build_plain_cnn(VOCAB_P)
    start = time.time()
    comb_model.fit(
        X_train_comb, y_train_comb,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_m_tokP, y_val_m),
        verbose=0
    )
    t_comb = time.time() - start
    y_prob = comb_model.predict(X_val_m_tokP, verbose=0).ravel()
    time_combined_scratch.append(t_comb)
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val (Pol+Med-Scratch Plain)")
    acc_combined_scratch.append(met['acc'])
    f1_combined_scratch.append(met['f1'])

# ======================
# 10. Plotting
# ======================
x_vals = [f*100 for f in fractions]
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(x_vals, acc_transfer, marker='o', label='Transfer+Adapter+EWC')
plt.plot(x_vals, acc_combined_scratch, marker='^', label='Pol+Med from scratch (Plain)')
plt.title('Medical Validation Accuracy vs. Percentage of Medical Train Data')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Time (seconds)
plt.subplot(1,2,2)
plt.plot(x_vals, time_transfer_total, marker='o', label='Transfer pretrain time')
plt.plot(x_vals, fine_turn, marker='o', label='Transfer finetune time')
plt.plot(x_vals, time_combined_scratch, marker='^', label='Pol+Med from scratch (Plain)')
plt.title('Training Time vs. Percentage of Medical Train Data')
plt.xlabel('Percentage of Medical Training Data (%)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================
# 11. Summary
# ======================
print("=== Summary (Medical Validation) ===")
for i, frac in enumerate(fractions):
    print(f"Medical data fraction: {int(frac*100)}%")
    print(f" -> Transfer+Adapter+EWC     Acc={acc_transfer[i]:.4f}  F1={f1_transfer[i]:.4f}  Time(pretrain)={time_transfer_total[i]:.2f}s  Time(finetune)={fine_turn[i]:.2f}s")
    print(f" -> Pol+Med from scratch (Plain) Acc={acc_combined_scratch[i]:.4f}  F1={f1_combined_scratch[i]:.4f}  Time={time_combined_scratch[i]:.2f}s")
    print("-"*60)
