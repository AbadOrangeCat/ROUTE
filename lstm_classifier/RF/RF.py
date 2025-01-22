# 导入所需的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 下载nltk资源


nltk.data.path.append('C:/Users/Eden/AppData/Roaming/nltk_data')  # 明确指定路径
nltk.download('punkt')

# 读取假新闻和真新闻的数据集
fake_df = pd.read_csv('../news/fake.csv')  # 假新闻数据集
true_df = pd.read_csv('../news/true.csv')  # 真新闻数据集


filtered_df = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]
fake_texts = filtered_df['text']

filtered_df = true_df[(true_df['subject'] == 'politicsNews') & (true_df['text'].str.len() >= 40)]
true_texts = filtered_df['text']

covid_fake_df = pd.read_csv('../covid/fakeNews.csv')  # 假医疗新闻数据集
covid_true_df = pd.read_csv('../covid/trueNews.csv')  # 真医疗新闻数据集

filtered_df = covid_fake_df[(covid_fake_df['Text'].str.len() >= 40)]
covid_fake_texts = filtered_df['Text']

filtered_df = covid_true_df[ (covid_true_df['Text'].str.len() >= 40)]
covid_true_texts = filtered_df['Text']


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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(df2['cleaned_text'], df2['label'], test_size=0.2, random_state=42)


# 使用TF-IDF向量化文本
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
X_test_tfidf2 = tfidf.transform(X_test2)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练分类器
rf_classifier.fit(X_train_tfidf, y_train)

# 进行预测
y_pred = rf_classifier.predict(X_test_tfidf)
y_pred2 = rf_classifier.predict(X_test_tfidf2)


# 打印分类报告和准确率
print("Class:")
print(classification_report(y_test, y_pred))
print("Acc:", accuracy_score(y_test, y_pred))

print("Covid Class:")
print(classification_report(y_test2, y_pred2))
print("Covid Acc:", accuracy_score(y_test2, y_pred2))