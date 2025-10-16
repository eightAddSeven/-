import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', 1000)       # 设置显示宽度
pd.set_option('display.max_colwidth', None) # 不限制列宽

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =========加载数据========
train_location="D:/A数学建模/house-prices-advanced-regression-techniques/train.csv"
train=pd.read_csv(train_location)


# ==============找到与目标关系最紧密的特征(数值列)================
# 假设 df 是你的训练数据，目标列为 'SalePrice'
target = 'SalePrice'

# 计算所有数值列与目标的相关系数
# corr() 函数返回一个 pandas DataFrame，其中包含所有数值列之间的相关系数矩阵
corr_with_target = train.corr(numeric_only=True)[target].sort_values(ascending=False)

# 输出前 10 个最相关的特征
print("与目标变量最相关的特征：")
print(corr_with_target.head(10))

# 可视化（相关性柱状图）
plt.figure(figsize=(8, 5))
corr_with_target.head(10).plot(kind='bar', color='teal')
plt.title("与目标变量相关性最高的特征")
plt.ylabel("相关系数")
plt.show()