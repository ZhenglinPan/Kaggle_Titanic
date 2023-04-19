import yaml
from tqdm import tqdm
import pandas as pd

from XGBoostClassifier import XGBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from main import read_data

import itertools

def run(config, data):
    
    x_train, y_train, x_val, y_val, x_test, y_test = data[0], data[1], data[2], data[3], data[4], data[5]
    
    classifier = XGBoostClassifier(config)
    classifier.fit(x_train, y_train, x_val, y_val)
    time = classifier.summary(draw=False)
    
    y_pred = classifier.predict(x_test)

    confusion = confusion_matrix(y_test, y_pred)    # v: pred, h:truth
    print("Confusion matrix: \n", confusion)
    acc = accuracy_score(y_test, y_pred)
    print("Predict accuracy: ", acc)
    
    pre = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    rec = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    spe = confusion[0][0] / (confusion[0][1] + confusion[0][0])
    f1 = 2*pre*rec / (pre + rec)
    
    return {"config": config, "time": time, "confusion": confusion, "acc": acc, "pre": pre, "spe": spe, "f1": f1}
    

if __name__ == "__main__":

    data_dir = r'E:\Kaggle\Titanic\data\train_new.csv'
    yaml_dir = r'E:\Kaggle\Titanic\src\models\XGBoost\config.yaml'
    save_dir = r'E:\Kaggle\Titanic\src\models\XGBoost\gridSearch.csv'
    
    x_train, y_train, x_val, y_val, x_test, y_test = read_data(data_dir)
    data = (x_train, y_train, x_val, y_val, x_test, y_test)
    
    with open(yaml_dir, 'r') as f:
        config_file = yaml.safe_load(f)
    configs = config_file['gridSearch']
    
    grid = []
    values = list(configs.values())
    for config in itertools.product(*values):
        grid.append({"epoch": config[0],
                     "Lambda": config[1],
                     "Gamma": config[2],
                     "cover": config[3],
                     "learning_rate": config[4],
                     "quantiles": config[5],
                     "tree_height": config[6],
                     "feature_ratio": config[7],
                     "sample_ratio": config[8]
                     })
    print("total config combinations: %d" % len(grid))
    
    history = {"epoch":[], 
               "Lambda":[], 
               "Gamma":[], 
               "cover":[], 
               "learning_rate":[], 
               "quantiles":[], 
               "tree_height":[], 
               "feature_ratio":[], 
               "sample_ratio":[],
               "time":[], 
               "confusion": [], 
               "acc": [], 
               "pre": [], 
               "spe": [], 
               "f1": [],
               }
    
    for i in tqdm(range(len(grid))):
        res = run(grid[i], data)
        for key in res.keys():
            if str(key) == "config":
                for config_key in res[key].keys():
                    history[config_key].append(res[key][config_key])
            else:
                history[key].append(res[key])
                
    df = pd.DataFrame.from_dict(history)
    df.to_csv(save_dir)
    
    
    
