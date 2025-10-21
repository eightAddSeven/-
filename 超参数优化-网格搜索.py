import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict  
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification

# 载入数据
data = pd.read_csv(r"filepath")
x_train = data.drop(columns=['target'])
y_train = data['target']

# 定义参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 创建随机森林分类器
model = RandomForestClassifier(random_state=42)

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

# 在训练数据上拟合网格搜索
grid_search.fit(x_train, y_train)

# 输出最佳参数和最佳得分
print("最佳参数: ", grid_search.best_params_)
print("最佳交叉验证得分: {:.4f}".format(grid_search.best_score_))

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)

# 预测训练集以评估模型性能
train_predictions = best_model.predict(x_train)
