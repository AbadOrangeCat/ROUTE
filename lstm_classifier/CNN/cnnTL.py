import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
from collections import Counter

# 下载nltk资源
nltk.download('punkt')
nltk.download('stopwords')

# 读取假新闻和真新闻的数据集
fake_df = pd.read_csv('../news/fake.csv')  # 假新闻数据集
true_df = pd.read_csv('../news/true.csv')  # 真新闻数据集

covid_fake_df = pd.read_csv('../covid/fakeNews.csv')  # 假医疗新闻数据集
covid_true_df = pd.read_csv('../covid/trueNews.csv')  # 真医疗新闻数据集

filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

filtered_df = true_df[(true_df['subject'] == 'politicsNews') & (true_df['text'].str.len() >= 40)]
true_texts = filtered_df['text']

filtered_df = covid_fake_df[(covid_fake_df['Text'].str.len() >= 40)]
covid_fake_texts = filtered_df['Text']

filtered_df = covid_true_df[(covid_true_df['Text'].str.len() >= 40)]
covid_true_texts = filtered_df['Text']

# 为假新闻和真新闻分别打上标签
fake_labels = [0] * len(fake_texts)  # 假新闻标签为0
true_labels = [1] * len(true_texts)  # 真新闻标签为1

covid_fake_labels  = [0] * len(covid_fake_texts)
covid_true_labels  = [1] * len(covid_true_texts)

# 创建数据框，将文本和标签组合
fake_data = pd.DataFrame({'text': fake_texts, 'label': fake_labels})
true_data = pd.DataFrame({'text': true_texts, 'label': true_labels})

covid_fake_data = pd.DataFrame({'text': covid_fake_texts, 'label': covid_fake_labels})
covid_true_data = pd.DataFrame({'text': covid_true_texts, 'label': covid_true_labels})

# 合并假新闻和真新闻的数据集
df = pd.concat([fake_data, true_data], ignore_index=True)
df2 = pd.concat([covid_fake_data, covid_true_data], ignore_index=True)

# 文本预处理函数
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()  # 使用split()进行简单分词
    words = [word for word in words if word.isalpha()]  # 移除非字母字符
    words = [word for word in words if word not in stop_words]  # 移除停用词
    return ' '.join(words)

# 预处理新闻内容
df['cleaned_text'] = df['text'].apply(preprocess_text)
df2['cleaned_text'] = df2['text'].apply(preprocess_text)

# **修改1：使用预训练的GloVe词向量**
# 下载GloVe嵌入
import os
import requests
import zipfile

glove_dir = './glove/'
if not os.path.exists(glove_dir):
    os.makedirs(glove_dir)

glove_zip = glove_dir + 'glove.6B.zip'
if not os.path.isfile(glove_zip):
    print("Downloading GloVe embeddings...")
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    r = requests.get(url, stream=True)
    with open(glove_zip, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # 解压缩
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extractall(glove_dir)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['cleaned_text'], df2['label'], test_size=0.2, random_state=42)

# 设定一些参数
MAX_NUM_WORDS = 20000  # 增加词汇表大小
MAX_SEQUENCE_LENGTH = 500  # 每篇文章限制最大单词数为500
EMBEDDING_DIM = 100  # 词嵌入维度

# **合并所有文本以拟合Tokenizer**
all_texts = pd.concat([X_train, X_train2, X_test, X_test2])

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(all_texts)

# 将所有数据集转换为序列
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train2_sequences = tokenizer.texts_to_sequences(X_train2)
X_test2_sequences = tokenizer.texts_to_sequences(X_test2)

# 填充序列
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_train2_padded = pad_sequences(X_train2_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test2_padded = pad_sequences(X_test2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# **创建嵌入矩阵**
print("Preparing embedding matrix...")
embeddings_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# **重新定义模型，使用预训练的词嵌入**
from tensorflow.keras.initializers import Constant

model = Sequential()
model.add(Embedding(input_dim=num_words,
                    output_dim=EMBEDDING_DIM,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))  # 词嵌入层不参与训练

# 卷积层+池化层
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))

# 再次卷积+池化
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))

# 展开特征
model.add(Flatten())

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 添加Dropout以防止过拟合
model.add(Dense(1, activation='sigmoid'))  # 输出层，二分类问题用sigmoid激活函数

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# **修改2：在政治新闻数据上训练模型**
model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_data=(X_test_padded, y_test))

# **修改3：在医疗新闻数据上微调，增加训练轮数和调整学习率**
# 检查医疗新闻数据的类别分布
counter = Counter(y_train2)
print("Medical News Training Data Distribution:", counter)

# **如果类别不平衡，使用class_weight**
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train2),
                                                  y=y_train2)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 重新编译模型，降低学习率
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 在医疗新闻数据上微调，增加训练轮数
model.fit(X_train2_padded, y_train2, epochs=200, batch_size=32, validation_data=(X_test2_padded, y_test2), class_weight=class_weights)

# 模型评估
y_pred = (model.predict(X_test_padded) > 0.5).astype("int32")
y_pred2 = (model.predict(X_test2_padded) > 0.5).astype("int32")

# 打印分类报告和准确率
print("Political News Report:")
print(classification_report(y_test, y_pred))
print("Political News Accuracy:", accuracy_score(y_test, y_pred))

print("\nMedical News Report:")
print(classification_report(y_test2, y_pred2))
print("Medical News Accuracy:", accuracy_score(y_test2, y_pred2))
