import numpy as np
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc
import joblib #jbolib模块
from sklearn.neighbors import LocalOutlierFactor


# 36 feature lable: 0 and 1

def svmpredictone():
    filepath = '../data/example36.csv'  # 预测 1

    data = pd.read_csv(filepath, header=None, sep=',')
    data = pd.DataFrame(data)
    data = data.dropna()
    x_test = data.iloc[:, 1:]
    y_test = data.iloc[:, 0]

    # load svm model
    modelpath = '../data/svmmodel_36.pkl'
    model = joblib.load(modelpath)
    #  predict
    y_test_pred = model.predict(x_test)
    # 测试集精确度
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print('SVM test Accuracy: {:.6f}'.format(test_accuracy))
    count_misclassified = (y_test != y_test_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    # 201 feature lable: 0 and 1
def svmpredicttwo():
    filepath = '../data/example201.csv'  # 预测 1

    data = pd.read_csv(filepath, header=None, sep=',')
    data = pd.DataFrame(data)
    data = data.dropna()
    x_test = data.iloc[:, 1:]
    y_test = data.iloc[:, 0]

    # load svm model
    modelpath = '../data/svmmodel_201.pkl'
    model = joblib.load(modelpath)
    #  predict
    y_test_pred = model.predict(x_test)
    # 测试集精确度
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print('SVM test Accuracy: {:.6f}'.format(test_accuracy))
    count_misclassified = (y_test != y_test_pred).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
if __name__ == '__main__':
    # 36 feature label : 0 and 1
    svmpredictone()
    # 201 feature l\abel : 0 and 1
    svmpredicttwo()



















