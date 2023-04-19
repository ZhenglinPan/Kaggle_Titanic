import pandas as pd
import numpy as np
import yaml

from sklearn.utils import shuffle
from XGBoostClassifier import XGBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def read_data(data_dir):
    data = pd.read_csv(data_dir)
    data = shuffle(data)

    TRAIN_N = int(data.shape[0] * 0.8)
    VAL_N = int(data.shape[0] * 0.1)
    TEST_N = data.shape[0] - TRAIN_N - VAL_N

    x_train, y_train = data.iloc[:TRAIN_N, 1:], data.iloc[:TRAIN_N, 0]
    x_val, y_val = data.iloc[TRAIN_N:TRAIN_N+VAL_N, 1:], data.iloc[TRAIN_N:TRAIN_N+VAL_N, 0]
    x_test, y_test = data.iloc[-TEST_N:, 1:], data.iloc[-TEST_N:, 0]

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":

    data_dir = r'E:\Kaggle\Titanic\data\train_new.csv'
    yaml_dir = r'E:\Kaggle\Titanic\src\models\XGBoost\config.yaml'
    
    with open(yaml_dir, 'r') as f:
        config_file = yaml.safe_load(f)
    config = config_file['config']

    x_train, y_train, x_val, y_val, x_test, y_test = read_data(data_dir)
    
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