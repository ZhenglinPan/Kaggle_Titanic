import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from XGBoostClassifier import XGBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == "__main__":

    data = pd.read_csv(r'E:\Kaggle\Titanic\data\train_new.csv')
    data = shuffle(data)

    TRAIN_N = int(data.shape[0] * 0.8)
    VAL_N = int(data.shape[0] * 0.1)
    TEST_N = data.shape[0] - TRAIN_N - VAL_N

    x_train, y_train = data.iloc[:TRAIN_N, 1:], data.iloc[:TRAIN_N, 0]
    x_val, y_val = data.iloc[TRAIN_N:TRAIN_N+VAL_N, 1:], data.iloc[TRAIN_N:TRAIN_N+VAL_N, 0]
    x_test, y_test = data.iloc[-TEST_N:, 1:], data.iloc[-TEST_N:, 0]
    
    config = {
        "epoch": 10,
        "Lambda": 1,
        "Gamma": 0,
        "cover": 0,
        "learning_rate": 0.3,
        "quantiles": 13, # normally 33
        "tree_height": 6,
        "feature_ratio": 0.25,
        "sample_ratio": 0.25,
    }
    
    classifier = XGBoostClassifier(config)
    classifier.fit(x_train, y_train, x_val, y_val)
    classifier.summary()
    
    y_pred = classifier.predict(x_test)

    confusion = confusion_matrix(y_test, y_pred)    # v: pred, h:truth
    print("Confusion matrix: \n", confusion)
    acc = accuracy_score(y_test, y_pred)
    print("Predict accuracy: ", acc)
    
    pre = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    rec = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    f1 = 2*pre*rec / (pre + rec)
    print("f1 score: ", f1)