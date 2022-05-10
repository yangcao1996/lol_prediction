import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('from_haoyang/LOL-predict-GBDT/LOL_main.csv')
print(data.keys())

epoch_num = 20

LR_model = MLPClassifier(learning_rate_init=0.009, hidden_layer_sizes=(100))
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
    y_predprob = LR_model.predict_proba(x_train)[:,1]
    print('===train===')
    #print(y_train.values)
    y_train_values = np.ascontiguousarray(y_train.values, dtype=np.float32)
    #print(y_pred)
#    print(y_train_values)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, y_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train_values, y_predprob))
    y_pred_ = LR_model.predict(x_test)
    y_predprob_ = LR_model.predict_proba(x_test)[:,1]
    print('===test===')
    y_test_values = np.ascontiguousarray(y_test.values, dtype=np.float32)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred_))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test_values, y_predprob_))

