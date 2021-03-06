import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('from_haoyang/LOL-predict-GBDT/LOL_main.csv')
print(data.keys())

epoch_num = 20

LR_model = LinearRegression()
for i in range(epoch_num):
    print('-----------------The %d epoch------------------' % i)
    target='t1_win'
    data = data.iloc[np.random.permutation(len(data))]
    # print(data.columns)
    x_columns = [x for x in data.columns if x not in [target]]

    x_train = data[x_columns][:-1001]
    y_train = data[target][:-1001]
    
    x_test = data[x_columns][-1000:]
    y_test = data[target][-1000:]
    
    # print(len(x_train))
    # print(len(x_test))
#    LR_model = LinearRegression()
    LR_model.fit(x_train, y_train)

    y_pred = LR_model.predict(x_train)
    #y_predprob = LR_model.predict(x_train)[:,1]
    print('===train===')
    #print(y_train.values)
    #y_train_values = np.ascontiguousarray(y_train.values, dtype=np.float32)
    #print(y_pred)
    #print(y_train.values)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, y_pred))
    #print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train_values, y_pred))
    y_pred_ = LR_model.predict(x_test)
    y_pred_[y_pred_ > 0.5] = 1
    y_pred_[y_pred_ < 0.5] = 0
    #y_predprob_ = LR_model.predict_proba(X_test)[:,1]
    print('===test===')
    #y_test_values = np.ascontiguousarray(y_test.values, dtype=np.float32)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred_))
    #print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test_values, y_pred_))
    


