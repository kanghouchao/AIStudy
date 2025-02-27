# 导入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分数据集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练 KNN 模型（K=3）
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. 预测
y_pred = knn.predict(X_test)

# 5. 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

# 6. 可视化数据（以两个特征为例）
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Dataset Visualization')
plt.show()
