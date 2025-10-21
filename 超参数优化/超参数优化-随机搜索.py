import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# 载入数据
data = pd.read_csv(r"filepath")
x_train = data.drop(columns=['target'])
y_train = data['target']

# 创建随机森林模型
classifier = RandomForestClassifier(n_jobs=-1)

# 创建随机搜索空间——这里的搜索空间更大更全面
param_grid = {
        "n_estimators": np.arange(100, 1500, 100), 
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"] 
}

# 进行随机搜索
random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    n_iter=50,  # 随机搜索的迭代次数
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# 在训练数据上拟合随机搜索
random_search.fit(x_train, y_train)
# 输出最佳参数和最佳得分
print("最佳参数: ", random_search.best_params_)
print("最佳交叉验证得分: {:.4f}".format(random_search.best_score_))

# 使用最佳参数训练最终模型
best_model = random_search.best_estimator_
best_model.fit(x_train, y_train)

# 预测训练集以评估模型性能
train_predictions = best_model.predict(x_train)