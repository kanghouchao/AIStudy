import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练决策树模型
dt = DecisionTreeClassifier(max_depth=4, random_state=42)  # 限制最大深度为3，防止过拟合
dt.fit(X_train, y_train)

# 4. 评估模型
accuracy = dt.score(X_test, y_test)
print(f'测试集准确率: {accuracy:.2f}')  # 计算模型在测试集上的准确率

# 5. 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()