
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertModel  # Using pre-trained BERT

# ==========================================================
# 1. 数据加载与预处理 (Data Loading and Preprocessing)
# ==========================================================
# 使用原始 fake.csv 与 true.csv 数据集
fake_df = pd.read_csv('../../news/fake.csv', encoding='utf-8')
real_df = pd.read_csv('../../news/true.csv', encoding='utf-8')
covid_fake_df = pd.read_csv('../../covid/fakeNews.csv', encoding='utf-8')
covid_real_df = pd.read_csv('../../covid/trueNews.csv', encoding='utf-8')

# 筛选“政治”文本 (subject 字段)作为训练集
politics_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
politics_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']

# 筛选“医疗”文本 (COVID 新闻)作为测试集
medical_fake = covid_fake_df[covid_fake_df['Text'].str.len() >= 40]['Text']
medical_real = covid_real_df[covid_real_df['Text'].str.len() >= 40]['Text']

# 合并并标注 (0 = 假新闻, 1 = 真新闻)
train_texts = pd.concat([politics_fake, politics_real]).reset_index(drop=True)
y_train = np.concatenate([
    np.zeros(len(politics_fake), dtype=int),
    np.ones(len(politics_real), dtype=int)
])
test_texts = pd.concat([medical_fake, medical_real]).reset_index(drop=True)
y_test = np.concatenate([
    np.zeros(len(medical_fake), dtype=int),
    np.ones(len(medical_real), dtype=int)
])

# 检查数据集规模
print(f"训练集样本数: {len(train_texts)},  真新闻比例: {y_train.mean():.2f}")
print(f"测试集样本数: {len(test_texts)},  真新闻比例: {y_test.mean():.2f}")

# ==========================================================
# 2. 模型定义：使用预训练 BERT (Model Definition using BERT)
# ==========================================================
# 定义最大序列长度（BERT 最大512，为平衡效率和信息，这里取256）
maxlen = 256
# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将文本转换为 BERT 输入格式（包含 input_ids, attention_mask, token_type_ids）
X_train_enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=maxlen)
X_test_enc  = tokenizer(list(test_texts),  truncation=True, padding=True, max_length=maxlen)

# 提取编码后的各个字段并转换为张量
X_train_input_ids = tf.constant(X_train_enc['input_ids'])
X_train_attention = tf.constant(X_train_enc['attention_mask'])
X_train_token_type= tf.constant(X_train_enc['token_type_ids'])
X_test_input_ids  = tf.constant(X_test_enc['input_ids'])
X_test_attention  = tf.constant(X_test_enc['attention_mask'])
X_test_token_type = tf.constant(X_test_enc['token_type_ids'])

# 构建BERT分类模型
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# BERT 输出包括 (last_hidden_states, pooled_output); 使用 pooled [CLS] 表示
input_ids = layers.Input(shape=(maxlen,), dtype=tf.int32, name='input_ids')
attention_mask = layers.Input(shape=(maxlen,), dtype=tf.int32, name='attention_mask')
token_type_ids = layers.Input(shape=(maxlen,), dtype=tf.int32, name='token_type_ids')
# 获取 BERT 输出
bert_outputs = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
cls_output = bert_outputs.pooler_output  # [CLS] 标记的池化输出
# 添加 Dropout 层防止过拟合
cls_output = layers.Dropout(0.3)(cls_output)
# 输出层：1个神经元，使用 Sigmoid 激活（预测为真新闻的概率）
predictions = layers.Dense(1, activation='sigmoid')(cls_output)
# 定义模型
model = Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=predictions)

# 编译模型：使用较低的学习率适合微调 BERT
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# ==========================================================
# 3. 在政治新闻训练集上训练模型 (Model Training on Training Set)
# ==========================================================
# 为防止输出过多日志信息，可以设置只显示每个epoch的简要结果
# (如果在交互式环境运行，可以启用下面这一行)
# tf.keras.callbacks.TerminateOnNaN()  # example callback if needed
history = model.fit(
    [X_train_input_ids, X_train_attention, X_train_token_type], y_train,
    batch_size=4, epochs=3, validation_split=0.1
)



# 4. 在医疗新闻测试集上进行预测 (Prediction on Test Set)
test_enc = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding='max_length',
    max_length=256,      # ← 必须和训练一致
    return_tensors='tf'
)
y_pred_prob = model.predict(
    {
        'input_ids':      test_enc['input_ids'],
        'attention_mask': test_enc['attention_mask'],
        'token_type_ids': test_enc['token_type_ids']
    },
    batch_size=4          # 和训练用的相同或更小
).flatten()


# 使用调整后的阈值分类：
# 如果概率>0.6 判断为真新闻；<0.4 判断为假新闻；介于其间视为不确定
high_thresh = 0.6
low_thresh  = 0.4
pred_labels_num = []
pred_categories = []
for p in y_pred_prob:
    if p >= high_thresh:
        pred_labels_num.append(1)
        pred_categories.append("真新闻")
    elif p <= low_thresh:
        pred_labels_num.append(0)
        pred_categories.append("假新闻")
    else:
        pred_labels_num.append(None)
        pred_categories.append("不确定")

# 5. 保存预测结果到 CSV 文件 (Save Predictions to CSV)
results_df = pd.DataFrame({
    "text": test_texts,
    "pred_probability": y_pred_prob,
    "pred_category": pred_categories
})
results_df.to_csv("predictions.csv", index=False, encoding='utf-8')

# 6. 计算评估指标并输出 (Evaluation Metrics)
total_samples = len(y_test)
# 总体准确率（将“不确定”视为错误预测）
correct_incl_uncertain = 0
for true_label, pred_label in zip(y_test, pred_labels_num):
    if pred_label is not None and pred_label == true_label:
        correct_incl_uncertain += 1
overall_accuracy = correct_incl_uncertain / total_samples if total_samples > 0 else 0.0

# 计算真新闻和假新闻的精确率、召回率、F1分数
tp_real = fp_real = fn_real = 0
tp_fake = fp_fake = fn_fake = 0
for true_label, pred_label in zip(y_test, pred_labels_num):
    # 真新闻（正类=1）
    if pred_label == 1 and true_label == 1:
        tp_real += 1
    if pred_label == 1 and true_label == 0:
        fp_real += 1
    if true_label == 1 and pred_label != 1:
        fn_real += 1
    # 假新闻（正类=0）
    if pred_label == 0 and true_label == 0:
        tp_fake += 1
    if pred_label == 0 and true_label == 1:
        fp_fake += 1
    if true_label == 0 and pred_label != 0:
        fn_fake += 1

precision_real = tp_real / (tp_real + fp_real) if (tp_real + fp_real) > 0 else 0.0
recall_real    = tp_real / (tp_real + fn_real) if (tp_real + fn_real) > 0 else 0.0
f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0.0
precision_fake = tp_fake / (tp_fake + fp_fake) if (tp_fake + fp_fake) > 0 else 0.0
recall_fake    = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0.0
f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0.0

# 排除“不确定”结果后的准确率（仅计算确定预测的准确率）
definite_total = sum(1 for label in pred_labels_num if label is not None)
definite_correct = sum(1 for true_label, pred_label in zip(y_test, pred_labels_num)
                       if pred_label is not None and pred_label == true_label)
accuracy_excluding_uncertain = definite_correct / definite_total if definite_total > 0 else 0.0

# 输出各项指标
print(f"总体准确率 (不确定视为错误): {overall_accuracy * 100:.2f}%")
print(f"真新闻 - 精确率: {precision_real * 100:.2f}%, 召回率: {recall_real * 100:.2f}%, F1: {f1_real * 100:.2f}%")
print(f"假新闻 - 精确率: {precision_fake * 100:.2f}%, 召回率: {recall_fake * 100:.2f}%, F1: {f1_fake * 100:.2f}%")
print(f"排除不确定结果后的准确率: {accuracy_excluding_uncertain * 100:.2f}%")

# 7. 绘制图表 (Visualization)
# 各类别预测结果的数量条形图
category_counts = {
    "真新闻": pred_categories.count("真新闻"),
    "假新闻": pred_categories.count("假新闻"),
    "不确定": pred_categories.count("不确定")
}
plt.figure(figsize=(5,4))
plt.bar(category_counts.keys(), category_counts.values(), color=['green', 'red', 'gray'])
plt.title("各类别预测结果数量")
plt.xlabel("预测类别")
plt.ylabel("新闻数量")
plt.savefig("category_counts.png")

# 模型预测为“真新闻”的概率分布直方图
real_probs = [p for p, cat in zip(y_pred_prob, pred_categories) if cat == "真新闻"]
plt.figure(figsize=(5,4))
plt.hist(real_probs, bins=10, range=(0.6, 1.0), color='green', edgecolor='black')
plt.title("预测为真新闻的概率分布")
plt.xlabel("预测为真新闻的概率")
plt.ylabel("频数")
plt.savefig("real_news_prob_dist.png")

print("图表已保存：category_counts.png, real_news_prob_dist.png")

# 8. 按类别输出统计表格 (Summary Table by Category)
df_results = pd.DataFrame({"true_label": y_test, "pred_category": pred_categories})
total = len(df_results)
def summarize_category(df, category):
    count = (df["pred_category"] == category).sum()
    pct = count / total * 100 if total > 0 else 0.0
    if category in ["真新闻", "假新闻"]:
        true_val = 1 if category == "真新闻" else 0
        tp = ((df["pred_category"] == category) & (df["true_label"] == true_val)).sum()
        pp = count  # predicted positives for this category
        ap = (df["true_label"] == true_val).sum()  # actual positives of this class
        precision = tp / pp * 100 if pp > 0 else 0.0
        recall = tp / ap * 100 if ap > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "数量": count,
            "百分比": f"{pct:.2f}%",
            "精确率": f"{precision:.2f}%",
            "召回率": f"{recall:.2f}%",
            "F1": f"{f1:.2f}%"
        }
    else:
        # 不确定类只输出数量和占比
        return {
            "数量": count,
            "百分比": f"{pct:.2f}%",
            "精确率": "--",
            "召回率": "--",
            "F1": "--"
        }

rows = []
for cat in ["真新闻", "假新闻", "不确定"]:
    stats = summarize_category(df_results, cat)
    stats["类别"] = cat
    rows.append(stats)
stats_df = pd.DataFrame(rows, columns=["类别", "数量", "百分比", "精确率", "召回率", "F1"])
print(stats_df.to_string(index=False, justify="left"))
