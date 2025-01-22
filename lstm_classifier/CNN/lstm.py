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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['cleaned_text'], df2['label'], test_size=0.2, random_state=42)

# 设定一些参数
MAX_NUM_WORDS = 5000  # 只考虑最常用的5000个单词
MAX_SEQUENCE_LENGTH = 500  # 每篇文章限制最大单词数为500
EMBEDDING_DIM = 100  # 词嵌入维度

# 使用Tokenizer将文本转换为整数序列
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_test_sequences2 = tokenizer.texts_to_sequences(X_test2)

# 填充序列，使得每篇文章的长度一致
X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test_padded2 = pad_sequences(X_test_sequences2, maxlen=MAX_SEQUENCE_LENGTH)

# LSTM模型定义
model = Sequential()

# 词嵌入层
model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

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

# 训练模型
model.fit(X_train_padded, y_train, epochs=5, batch_size=64, validation_data=(X_test_padded, y_test))

# 模型评估
y_pred = (model.predict(X_test_padded) > 0.5).astype("int32")
y_pred2 = (model.predict(X_test_padded2) > 0.5).astype("int32")

# 打印分类报告和准确率
print("报告:")
print(classification_report(y_test, y_pred))
print("准确率:", accuracy_score(y_test, y_pred))

print("Covid数据集结果:")
print(classification_report(y_test2, y_pred2))
print("Covid准确率:", accuracy_score(y_test2, y_pred2))
