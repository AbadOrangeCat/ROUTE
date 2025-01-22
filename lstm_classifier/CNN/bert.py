
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.optimizers import Adam

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

# 划分政治新闻的训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
# 划分医疗新闻的训练集和测试集
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['cleaned_text'], df2['label'], test_size=0.2, random_state=42)

# 加载预训练的BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将文本编码为BERT所需的格式
def encode_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors='tf')

# 编码政治新闻数据
X_train_enc = encode_texts(X_train)
X_test_enc = encode_texts(X_test)

# 编码医疗新闻数据
X_train2_enc = encode_texts(X_train2)
X_test2_enc = encode_texts(X_test2)

# **修改1：将标签包含在输入字典中**
train_dataset = tf.data.Dataset.from_tensor_slices({**X_train_enc, 'labels': y_train.values}).shuffle(10000).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices({**X_test_enc, 'labels': y_test.values}).batch(8)

train2_dataset = tf.data.Dataset.from_tensor_slices({**X_train2_enc, 'labels': y_train2.values}).shuffle(10000).batch(8)
test2_dataset = tf.data.Dataset.from_tensor_slices({**X_test2_enc, 'labels': y_test2.values}).batch(8)

# 加载预训练的BERT模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 编译模型
optimizer = Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# **第一步：在政治新闻数据上微调BERT模型**
print("在政治新闻数据集上微调...")
model.fit(train_dataset, epochs=2, validation_data=test_dataset)

# **第二步：在医疗新闻数据上进一步微调**
print("在医疗新闻数据集上微调...")
model.compile(optimizer=Adam(learning_rate=2e-5), loss=model.compute_loss, metrics=['accuracy'])
model.fit(train2_dataset, epochs=3, validation_data=test2_dataset)

# **模型评估**

# 政治新闻测试集评估
y_pred = []
y_true = []
for batch in test_dataset:
    logits = model(batch)['logits']
    predictions = tf.argmax(logits, axis=-1).numpy()
    y_pred.extend(predictions)
    y_true.extend(batch['labels'].numpy())

print("政治新闻测试集结果:")
print(classification_report(y_true, y_pred))
print("政治新闻准确率:", accuracy_score(y_true, y_pred))

# 医疗新闻测试集评估
y_pred2 = []
y_true2 = []
for batch in test2_dataset:
    logits = model(batch)['logits']
    predictions = tf.argmax(logits, axis=-1).numpy()
    y_pred2.extend(predictions)
    y_true2.extend(batch['labels'].numpy())

print("医疗新闻测试集结果:")
print(classification_report(y_true2, y_pred2))
print("医疗新闻准确率:", accuracy_score(y_true2, y_pred2))
