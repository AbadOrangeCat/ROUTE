import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# 下载nltk资源
nltk.download('punkt')
nltk.download('stopwords')

# 读取假新闻和真新闻的数据集
fake_df = pd.read_csv('../news/fake.csv')  # 假新闻数据集
true_df = pd.read_csv('../news/true.csv')  # 真新闻数据集

covid_fake_df = pd.read_csv('../covid/fakeNews.csv')  # 假医疗新闻数据集
covid_true_df = pd.read_csv('../covid/trueNews.csv')  # 真医疗新闻数据集

# 数据过滤和选择
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

covid_fake_labels = [0] * len(covid_fake_texts)
covid_true_labels = [1] * len(covid_true_texts)

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

# **修改1：合并所有文本以拟合Tokenizer**
# 合并政治新闻和医疗新闻的文本
all_texts = pd.concat([df['cleaned_text'], df2['cleaned_text']], ignore_index=True)

# 设定一些参数
MAX_NUM_WORDS = 20000  # 增加词汇表大小，覆盖更多词汇
MAX_SEQUENCE_LENGTH = 500  # 每篇文章限制最大单词数为500
EMBEDDING_DIM = 100  # 词嵌入维度

# 使用Tokenizer将文本转换为整数序列
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(all_texts)

# 将所有数据集转换为序列
X_sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
X2_sequences = tokenizer.texts_to_sequences(df2['cleaned_text'])

# 填充序列，使得每篇文章的长度一致
X_padded = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X2_padded = pad_sequences(X2_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# 划分政治新闻的训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_padded, df['label'], test_size=0.2, random_state=42)
# 划分医疗新闻的训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_padded, df2['label'], test_size=0.2, random_state=42)

# **修改2：使用预训练的GloVe词向量**
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

# 加载GloVe嵌入
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

# LSTM模型定义
from tensorflow.keras.initializers import Constant
model = Sequential()

# **使用预训练的词嵌入层**
model.add(Embedding(input_dim=num_words,
                    output_dim=EMBEDDING_DIM,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))  # 不训练嵌入层

# LSTM层
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # 输出层，二分类问题用sigmoid激活函数

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# **修改3：在政治新闻数据上训练模型**
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# **修改4：在医疗新闻数据上进行微调**
# 检查医疗新闻训练数据的类别分布
counter = Counter(y_train2)
print("Medical News Training Data Distribution:", counter)

# 处理类别不平衡
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train2),
                                                  y=y_train2)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 重新编译模型，降低学习率
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 在医疗新闻数据上微调
model.fit(X_train2, y_train2, epochs=100, batch_size=32, validation_data=(X_test2, y_test2), class_weight=class_weights)

# 模型评估
y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_pred2 = (model.predict(X_test2) > 0.5).astype("int32")

# 打印分类报告和准确率
print("政治新闻测试集结果:")
print(classification_report(y_test, y_pred))
print("政治新闻准确率:", accuracy_score(y_test, y_pred))

print("医疗新闻测试集结果:")
print(classification_report(y_test2, y_pred2))
print("医疗新闻准确率:", accuracy_score(y_test2, y_pred2))
