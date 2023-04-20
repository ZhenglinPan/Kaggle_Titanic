import os
import time
import yaml
import itertools
import pandas as pd
from tqdm import tqdm

import threading
import queue

from XGBoostClassifier import XGBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from main import read_data

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


def run(config, data):
    
    x_train, y_train, x_val, y_val, x_test, y_test = data[0], data[1], data[2], data[3], data[4], data[5]
    
    start_time = time.time()
    classifier = XGBoostClassifier(config)
    classifier.fit(x_train, y_train, x_val, y_val)
    classifier.summary(draw=False)
    time_cost = time.time() - start_time
    
    y_pred = classifier.predict(x_test)

    confusion = confusion_matrix(y_test, y_pred)    # v: pred, h:truth
    acc = accuracy_score(y_test, y_pred)
    pre = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    rec = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    spe = confusion[0][0] / (confusion[0][1] + confusion[0][0])
    f1 = 2*pre*rec / (pre + rec)
    
    return {"time": time_cost, "confusion": confusion, "acc": acc, "pre": pre, "spe": spe, "f1": f1}
    

def load_checkpoint(save_dir, history) -> list:
    if os.path.exists(save_dir):    # load checkpoint if there exists checkpoint
        checkpoint = pd.read_csv(save_dir, index_col = [0])  
        keys = checkpoint.columns
        for key in keys:
            history[key] = list(checkpoint[key])
        
    return history


def get_configs(save_dir):
    configs = []
    keys = list(history.keys())[:9]
    if os.path.exists(save_dir):
        checkpoint = pd.read_csv(save_dir)
        for i in range(checkpoint.shape[0]):
            config = []
            for key in keys:
                config.append(list(checkpoint[key])[i])
            configs.append((tuple(config)))
    return configs


def convert(h):
    for key in h.keys():
        h[key] = [h[key]]

    return h


def thread_grid_search(grid, histories, data):
    while True:
        config = grid.get()
        print(threading.current_thread().name, f"config on run: {config}")
        
        res = run(config, data)
        histories.put({**config, **res})


def thread_save_history(histories, save_dir):
    while True:
        h = histories.get()
        df = pd.DataFrame.from_dict(convert(h))
        df.to_csv(save_dir, index=False, header=False, mode='a')
        

if __name__ == '__main__':

    data_dir = './data/train_new.csv'
    yaml_dir = './src/models/XGBoost/config.yaml'
    save_dir = './src/models/XGBoost/gridSearch.csv'

    x_train, y_train, x_val, y_val, x_test, y_test = read_data(data_dir)
    data = (x_train, y_train, x_val, y_val, x_test, y_test)
    
    with open(yaml_dir, 'r') as f:
        config_file = yaml.safe_load(f)
    configs = config_file['gridSearch']
    
    history = load_checkpoint(save_dir, history)
    grid_done = get_configs(save_dir)
    
    if len(grid_done) == 0 : 
        df = pd.DataFrame.from_dict(history)
        df.to_csv(save_dir, index=False)
    
    grid = queue.Queue()
    histories = queue.Queue()
    
    values = list(configs.values())
    for config in itertools.product(*values):
        if config not in grid_done:
            grid.put({"epoch": config[0],
                        "Lambda": config[1],
                        "Gamma": config[2],
                        "cover": config[3],
                        "learning_rate": config[4],
                        "quantiles": config[5],
                        "tree_height": config[6],
                        "feature_ratio": config[7],
                        "sample_ratio": config[8]
                        })
        else:
            continue
    print("total config combinations: %d" % grid.qsize())
    
    for idx in range(10):
        t = threading.Thread(target=thread_grid_search, args=(grid, histories, data), name=f"--thread_search{idx}--")
        t.start()
    
    for idx in range(2):
        t = threading.Thread(target=thread_save_history, args=(histories, save_dir), name=f"--thread_save: {idx}--")
        t.start()

