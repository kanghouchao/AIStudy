from sklearn.datasets import load_iris

# 1️⃣ 加载数据集
iris = load_iris()

# 2️⃣ 特征数据 (150 × 4)
X = iris.data
print("特征数据：")
print(X[:5])  # 打印前 5 行

# 3️⃣ 目标数据 (150,)
y = iris.target
print("目标数据：")
print(y[:5])  # 打印前 5 个标签

# 4️⃣ 特征名称
print("特征名称：", iris.feature_names)

# 5️⃣ 目标标签名称
print("目标标签名称：", iris.target_names)

# 6️⃣ 数据集描述
print("数据集描述：")
print(iris.DESCR)