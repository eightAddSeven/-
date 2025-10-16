import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 载入数据
df_train=pd.read_csv("")
df_test=pd.read_csv("")

# ==============分析数据===============
df_train.describe()  #查看数据集里的最大值、平均值等等

# 画图
plt.figure(figsize=(9,8)) #创建画布
sns.histplot(df_train['某个变量'])
plt.show()

# 对训练集进行分割
x_train=df_train.drop(['目标特征'],axis=1)
y_train=df_train['目标特征']

# 查看数字集列
df_num=df_train.select_dtypes(include=['float64','int64'])
df_num.hist(figsize=(16,20),bins=50,xlabelsize=8, ylabelsize=8)
plt.tight_layout()  # 自动调整子图间距
plt.show()  # python脚本中，必须使用show显式显示图形，不同于notebook

# 分析某个特征与目标的关系
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)  #例如分析总体质量与房价的关系并画出箱子线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='OverallQual', y="SalePrice", data=data)
plt.show()

data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')  #散点图

#====correlation matrix====
corrmat = df_train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, vmax=.8, square=True)  #热力图

# =================处理数据===================

# 数据准备：填充缺失值
# 添加代码 添加Markdown
# 有几种经典的方法——删除包含此类数据的行、用平均值填充、根据数据特性用合理的值填充，或者构建例如随机森林模型并迭代地填充缺失值。首先，我们来研究一下缺失值。
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) #求出占比
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20) #这里展示缺失值数量最大的前20，根据题目可以改变参数

# 删除某些列
x_train = x_train.drop((missing_data[missing_data['Total'] > 81]).index,axis=1) #axis=1:按列删除
# 填充
x_train = x_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
x_train.isnull().sum().max() # 检查是否还有缺失值

# 处理测试集中的缺失值 - 不能删除行
df_test.info()

# 选择特征值类型为object
x_train_object=x_train.select_dtypes(include='object').columns #这个函数返回的是object的列索引


#特征编码——标签编码（带有一定的顺序性）
from sklearn.preprocessing import LabelEncoder
cols = x_train.select_dtypes(include='object').columns

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(x_train[c].values)) 
    x_train[c] = lbl.transform(list(x_train[c].values))
    df_test[c] = lbl.transform(list(df_test[c].values))

print('Shape all_data: {}'.format(x_train.shape))

# 当测试集中有训练集中没有的特征时需要进行对齐，否则模型报错或结果错误
# 如果训练集的数据清洗中有删除行的行为，测试集的清洗就不能直接调用，因为测试集的行数必须完整

# 在房价预测中我使用过的对齐代码
# X_test = data_clean(test_raw, is_train=False, reference_columns=train_columns)  # 用训练集列对齐

