import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

from sklearn.model_selection import train_test_split

import pickle

# Load fake and real news datasets
fake_df = pd.read_csv('../news/fake.csv')  # Fake news dataset
real_df = pd.read_csv('../news/true.csv')  # Real news dataset

# For medical news, we're reusing the same datasets as placeholders
covid_fake_df = pd.read_csv('../news/fake.csv')  # Fake medical news dataset
covid_real_df = pd.read_csv('../news/true.csv')  # Real medical news dataset

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

# Combine political news texts and labels
policy_texts = pd.concat([fake_texts, real_texts])
policy_labels = np.concatenate([np.zeros(len(fake_texts)), np.ones(len(real_texts))])  # 0 for fake, 1 for real news

# Combine medical news texts and labels
medical_texts = pd.concat([covid_fake_texts, covid_real_texts])
medical_labels = np.concatenate(
    [np.zeros(len(covid_fake_texts)), np.ones(len(covid_real_texts))])  # 0 for fake, 1 for real news

# Initialize Tokenizer and fit on all texts
all_texts = pd.concat([policy_texts, medical_texts])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(all_texts)

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert texts to sequences
policy_sequences = tokenizer.texts_to_sequences(policy_texts)
medical_sequences = tokenizer.texts_to_sequences(medical_texts)

# Pad sequences
maxlen = 1000  # Maximum sequence length
policy_data = pad_sequences(policy_sequences, maxlen=maxlen)
medical_data = pad_sequences(medical_sequences, maxlen=maxlen)

# Convert labels to numpy arrays
policy_labels = np.array(policy_labels)
medical_labels = np.array(medical_labels)

# Split political news dataset into training and validation sets
X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(policy_data, policy_labels, test_size=0.2, random_state=42)

# Split medical news dataset into training and validation sets
X_train_m_full, X_val_m, y_train_m_full, y_val_m = train_test_split(medical_data, medical_labels, test_size=0.2,
                                                                    random_state=42)

# Build CNN model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_dim = 100  # Embedding dimensions


def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Train the model on political news dataset
model = create_model()
history = model.fit(X_train_p, y_train_p, epochs=5, batch_size=64, validation_data=(X_val_p, y_val_p))

# Save the model weights
model.save_weights('policy_news_cnn_model.h5')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Plot accuracy and loss curves for political news model
plt.figure(figsize=(12, 4))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Political News Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Political News Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model on validation data before transfer learning
print("Evaluation on Medical News Validation Data before Transfer Learning:")
loss_m, accuracy_m = model.evaluate(X_val_m, y_val_m, verbose=0)
print(f"Medical News Validation Accuracy: {accuracy_m:.4f}")

print("Evaluation on Political News Validation Data before Transfer Learning:")
loss_p, accuracy_p = model.evaluate(X_val_p, y_val_p, verbose=0)
print(f"Political News Validation Accuracy: {accuracy_p:.4f}")

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Recalculate vocab_size
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size

# Define different fractions of the medical dataset to use
fractions = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # 1%, 5%, 10%, 20%, 50%, 100%
medical_accuracies = []

for frac in fractions:
    print(f"\nUsing {int(frac * 100)}% of the medical training data:")
    # Sample a fraction of the training data
    if frac < 1.0:
        X_train_m, _, y_train_m, _ = train_test_split(X_train_m_full, y_train_m_full, train_size=frac, random_state=42)
    else:
        X_train_m = X_train_m_full
        y_train_m = y_train_m_full

    # Create a new model to ensure fair comparison
    model = create_model()

    # Load the weights from the model trained on political news
    model.load_weights('policy_news_cnn_model.h5')

    # Fine-tune the model on the sampled medical news dataset
    history_m = model.fit(X_train_m, y_train_m, epochs=5, batch_size=64, validation_data=(X_val_m, y_val_m), verbose=0)

    # Evaluate the model on the medical news validation data
    loss_m, accuracy_m = model.evaluate(X_val_m, y_val_m, verbose=0)
    medical_accuracies.append(accuracy_m)
    print(f"Medical News Validation Accuracy: {accuracy_m:.4f}")

    # Evaluate the model on the political news validation data
    loss_p, accuracy_p = model.evaluate(X_val_p, y_val_p, verbose=0)
    print(f"Political News Validation Accuracy: {accuracy_p:.4f}")

# Plot medical news validation accuracy against data fraction
plt.figure()
plt.plot([f * 100 for f in fractions], medical_accuracies, marker='o')
plt.title('Medical News Validation Accuracy vs. Training Data Size')
plt.xlabel('Percentage of Medical Training Data Used (%)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()
