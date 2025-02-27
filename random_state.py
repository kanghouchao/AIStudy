from sklearn.model_selection import train_test_split
import numpy as np

# 创建示例数据
X = np.array(range(10)).reshape(-1, 1)  # 特征数据 [0,1,2,3,4,5,6,7,8,9]
y = np.array(range(10))  # 标签数据 [0,1,2,3,4,5,6,7,8,9]

# 拆分数据（第一次）
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# 拆分数据（第二次）
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

# 比较结果
print("第一次拆分的测试集:", X_test1.flatten())
print("第二次拆分的测试集:", X_test2.flatten())

# 输出：
# 第一次拆分的测试集: [8 1]
# 第二次拆分的测试集: [8 1]  （完全相同）