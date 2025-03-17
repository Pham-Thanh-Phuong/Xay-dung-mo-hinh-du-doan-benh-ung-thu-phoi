# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Khởi tạo SparkSession
spark = SparkSession.builder.appName("LungCancerPrediction").getOrCreate()
# Hiển thị link Spark UI
spark_url = spark.sparkContext.uiWebUrl
print(f"🔗 Spark UI Link: {spark_url}")
# Đọc dữ liệu CSV
df = spark.read.csv("lung_cancer_data.csv", header=True, inferSchema=True)

# Xử lý dữ liệu thiếu
df = df.dropna()

indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_Index").fit(df)
    for col_name in ["Smoking_History", "Tumor_Location", "Stage"]
]

for indexer in indexers:
    df = indexer.transform(df)

df = df.withColumn(
    "Age_Class",
    when(col("Age").between(30, 39), 0)
    .when(col("Age").between(40, 49), 1)
    .when(col("Age").between(50, 59), 2)
    .when(col("Age").between(60, 69), 3)
    .when(col("Age").between(70, 79), 4)
    .otherwise(5)
)

df = df.withColumn(
    "Survival_Class",
    when(col("Survival_Months") < 24, 0)
    .when(col("Survival_Months").between(24, 47), 1)
    .when(col("Survival_Months").between(48, 71), 2)
    .when(col("Survival_Months").between(72, 95), 3)
    .when(col("Survival_Months").between(96, 119), 4)
    .otherwise(5)
)
df = df.withColumn("Survival", (col("Survival_Months") >= 60).cast("integer"))

# Biến đổi dữ liệu thành vector đặc trưng
feature_cols = ["Smoking_History_Index", "Tumor_Location_Index", "Stage_Index", "Age_Class", "Survival_Class"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Chuyển dữ liệu PySpark thành Pandas
df_pandas = df.select(feature_cols + ["Survival", "Tumor_Size_mm"]).toPandas()

# Mã hóa nhãn dữ liệu
le = LabelEncoder()
df_pandas['Smoking_History_Index'] = le.fit_transform(df_pandas['Smoking_History_Index'])
df_pandas['Tumor_Location_Index'] = le.fit_transform(df_pandas['Tumor_Location_Index'])
df_pandas['Stage_Index'] = le.fit_transform(df_pandas['Stage_Index'])
df_pandas['Survival_Class'] = le.fit_transform(df_pandas['Survival_Class'])

# Tách tập dữ liệu train/test
X = df_pandas[feature_cols]
y = df_pandas["Survival"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình cây quyết định
model = DecisionTreeClassifier(max_depth=4, min_samples_split=50, min_samples_leaf=25)
model.fit(X_train, y_train)

# Vẽ cây quyết định
def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(14, 8))
    plot_tree(
        model, 
        feature_names=feature_names, 
        class_names=["Not Survived", "Survived"], 
        filled=True, 
        rounded=True, 
        proportion=False,  # Hiển thị số mẫu thực tế
        impurity=False,  # Ẩn giá trị impurity
        fontsize=10
    )
    plt.title("Decision Tree", fontsize=14)
    plt.show()

plot_decision_tree(model, X.columns)

stage_mapping = {0: "Stage I", 1: "Stage II", 2: "Stage III", 3: "Stage IV"}  # Định nghĩa ánh xạ

df_pandas["Stage_Index"] = df_pandas["Stage_Index"].map(stage_mapping)
stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]  # Định nghĩa thứ tự

df_pandas["Stage_Index"] = pd.Categorical(df_pandas["Stage_Index"], categories=stage_order, ordered=True)
stage_counts = df_pandas['Stage_Index'].value_counts().reindex(stage_order)
plt.figure(figsize=(8, 6))
stage_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Distribution of Cancer Stages')
plt.xlabel('Cancer Stage')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


smoking_mapping = {0: "Current Smoker", 1: "Never Smoked ", 2: "Former Smoker"}  # Định nghĩa ánh xạ

df_pandas["Smoking_History_Index"] = df_pandas["Smoking_History_Index"].map(smoking_mapping)
smoking_order = ["Current Smoker", "Never Smoked ", "Former Smoker"]  # Định nghĩa thứ tự mong muốn

df_pandas["Smoking_History_Index"] = pd.Categorical(df_pandas["Smoking_History_Index"], categories=smoking_order, ordered=True)
smoking_counts = df_pandas['Smoking_History_Index'].value_counts().reindex(smoking_order)
plt.figure(figsize=(8, 6))
smoking_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


bins = pd.qcut(df_pandas['Tumor_Size_mm'], 4)
df_pandas.groupby(bins)['Tumor_Size_mm'].hist(bins=10, edgecolor='black', figsize=(10, 6))
plt.suptitle('Histograms of Tumor Size for Each Quantile Bin', fontsize=16)
plt.xlabel('Tumor Size (mm)')
plt.ylabel('Frequency')
plt.show()

maxdepths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
trainAcc = []
testAcc = []

for depth in maxdepths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    trainAcc.append(accuracy_score(y_train, clf.predict(X_train)))
    testAcc.append(accuracy_score(y_test, clf.predict(X_test)))

plt.plot(maxdepths, trainAcc, 'ro-', label='Training Accuracy')
plt.plot(maxdepths, testAcc, 'bv--', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Chạy GridSearchCV để tìm mô hình tốt nhất
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'ccp_alpha': [0.0, 0.01, 0.05]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Gán mô hình tốt nhất từ GridSearchCV
best_model = grid_search.best_estimator_

# In kết quả tốt nhất
print(f"Best Model Accuracy: {best_model.score(X_test, y_test) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, best_model.predict(X_test)))

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Tạo ma trận nhầm lẫn
labels = [0, 1]  # 0: Not Survived, 1: Survived
conf_matrix = confusion_matrix(y_test, best_model.predict(X_test), labels=labels)

# Chuyển ma trận thành DataFrame
conf_matrix_df = pd.DataFrame(
    conf_matrix, 
    columns=['Predicted Not Survived', 'Predicted Survived'], 
    index=['True Not Survived', 'True Survived']
)

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1, square=True)

# Cập nhật tiêu đề và nhãn trục
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix for Cancer Survival Classification', fontsize=14)

# Hiển thị biểu đồ
plt.show()
