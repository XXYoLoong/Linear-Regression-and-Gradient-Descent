import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
import os
import tkinter

print(tkinter.TkVersion)

file_path = r'F:\工程训练\2_12.13\ex1data1.txt'
print("文件是否存在:", os.path.exists(file_path))

# 设置绘图风格
sns.set(context="notebook", style="whitegrid", palette="dark")

# 读取数据文件 ex1data1.txt，并为列命名
df = pd.read_csv(r'F:\工程训练\2_12.13\ex1data1.txt', names=['population', 'profit'])

# 查看数据前五行
print(df.head())

# 打印数据的基本信息
df.info()

# 打印统计信息（均值、标准差等）
print(df.describe())

# 绘制散点图展示数据分布
sns.lmplot(x='population', y='profit', data=df, height=6, fit_reg=False)
plt.show()

# 定义提取特征矩阵的函数
def get_X(df):
    """添加偏置项并返回特征矩阵"""
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # 偏置项
    data = pd.concat([ones, df], axis=1)  # 将偏置项和特征合并
    return data.iloc[:, :-1].values  # 返回所有列（去掉最后一列）

# 定义提取标签向量的函数
def get_y(df):
    """假定最后一列为目标标签"""
    return np.array(df.iloc[:, -1])  # 返回最后一列数据

# 特征归一化函数
def normalize_feature(df):
    """对每一列进行归一化处理"""
    return df.apply(lambda column: (column - column.mean()) / column.std())

# 提取特征和标签
data = df
X = get_X(data)
print(X.shape, type(X))  # 查看特征矩阵的形状和类型

y = get_y(data)
print(y.shape, type(y))  # 查看标签向量的形状和类型

# 初始化参数 theta
theta = np.zeros(X.shape[1])  # 参数个数等于特征个数
print(theta)

# 定义线性回归的代价函数
def lr_cost(theta, X, y):
    """计算线性回归的代价"""
    m = X.shape[0]  # 样本数
    inner = X @ theta - y  # 预测值与实际值的误差
    square_sum = inner.T @ inner  # 误差平方和
    cost = square_sum / (2 * m)  # 平均代价
    return cost

# 打印初始代价
print(lr_cost(theta, X, y))

# 定义梯度计算函数
def gradient(theta, X, y):
    """计算代价函数的梯度"""
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)  # 梯度公式
    return inner / m

# 定义批量梯度下降函数
def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    """执行批量梯度下降优化"""
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 创建 theta 的副本
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)  # 更新参数
        cost_data.append(lr_cost(_theta, X, y))  # 记录代价
    return _theta, cost_data

# 设置梯度下降的参数
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

# 打印最终的参数值和代价
print(final_theta)
print(lr_cost(final_theta, X, y))

# 可视化代价的变化过程
plt.plot(np.arange(len(cost_data)), cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Cost over Epochs')
plt.show()

# 使用 sklearn 的线性回归模型进行拟合
model = linear_model.LinearRegression()
model.fit(X, y)

# 绘制预测结果
x = X[:, 1]
f = model.predict(X).flatten()

plt.scatter(X[:, 1], y, label='Training Data')
plt.plot(x, f, 'r', label='Prediction')
plt.legend(loc=2)
plt.show()

# 读取 ex1data2.txt 数据文件并查看前五行
raw_data = pd.read_csv(r'F:\工程训练\2_12.13\ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(raw_data.head())

# 对新数据进行归一化处理
data = normalize_feature(raw_data)
print(data.head())

# 提取特征和标签
X = get_X(data)
print(X.shape, type(X))
y = get_y(data)
print(y.shape, type(y))

# 设置学习率和迭代次数，执行梯度下降
alpha = 0.01
theta = np.zeros(X.shape[1])
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)

# 可视化代价变化过程
plt.plot(np.arange(len(cost_data)), cost_data)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.title('Cost over Epochs with Multiple Features')
plt.show()

# 3D 可视化线性回归结果
model = linear_model.LinearRegression()
model.fit(X, y)

f = model.predict(X).flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 1], X[:, 2], f, 'r', label='Prediction')
ax.scatter(X[:, 1], X[:, 2], y, label='Training Data')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.legend()
plt.show()

# 测试不同学习率下的代价变化
base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))
print(candidate)

epoch = 50
fig, ax = plt.subplots(figsize=(8, 8))

for alpha in candidate:
    _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(epoch+1), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('cost', fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('Learning Rate Comparison', fontsize=12)
plt.show()

# 定义正规方程求解方法
def normalEqn(X, y):
    """通过正规方程计算参数"""
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # 求解公式
    return theta

# 计算最终参数并可视化
final_theta2 = normalEqn(X, y)
print(final_theta2)

f = final_theta2[0] + final_theta2[1] * X[:, 1] + final_theta2[2] * X[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 1], X[:, 2], f, 'r', label='Prediction')
ax.scatter(X[:, 1], X[:, 2], y, label='Training Data')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
plt.legend()
plt.show()
