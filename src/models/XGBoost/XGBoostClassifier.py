import numpy as np
import random
from DTtree import BTtreeNode
from tqdm import tqdm
from matplotlib import pyplot as plt

class XGBoostClassifier():

    forest = []
    
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    def __init__(self, config):
        self.EPOCH = config["epoch"]
        self.Lambda = config["Lambda"]
        self.Gamma = config["Gamma"]
        self.lr = config["learning_rate"]
    
    def fit(self, x_train, y_train, x_val, y_val):
        
        prob_train = np.zeros((x_train.shape[0], 1)) * 0.5
        res_train = y_train - prob_train
        
        for epoch in range(self.EPOCH):            
            x_rand, res_rand, prob_rand = self.__get_partial_data(x_train, res_train, prob_train)
            tree = self.__grow_tree(x_rand, res_rand, prob_rand, height=6)
            self.forest.append(tree)
            
            prob_train = self.__update(tree, x_train, prob_train)
            res_train = y_train - prob_train
            
            pred_train = ((prob_train > 0.5) * 1).reshape(-1, 1)
            self.train_loss.append(self.__cross_entropy_loss(y_train, prob_train))
            self.train_acc.append(np.mean(y_train, pred_train))
            
            prob_val = self.predict(x_val, y_val, ret_prob=True)
            pred_val = self.predict(x_val, y_val, ret_prob=False)
            self.val_loss.append(self.__cross_entropy_loss(y_val, prob_val))
            self.val_acc.append(np.mean(y_val, pred_val))
            

    def __cross_entropy_loss(self, truth, pred):
        return (- 1 / truth.shape[1]) * np.sum(truth * np.log(pred + 1e-16))
        

    def __update(self, tree, X, pre_prob):
        """Update the prob for X on a tree given previous prob"""
        prob = pre_prob.copy()
        for row in range(X.shape[0]):   # for each input x
            x = X.iloc[row, :]
            p = prob[row]
            log_odds = np.log((p / 1 - p) + 1e-16) + self.lr * self.__reach_leaf(tree, x)
            prob[row] = 1 / (1 + np.exp(log_odds))
        
        return prob
    
    def predict(self, X, ret_prob=True):
        """Predict the probability of X on forest with initial prob"""
        prob = np.zeros((X.shape[0], 1)) * 0.5
        for row in range(X.shape[0]):   # for each input x
            x = X.iloc[row, :]
            p = prob[row]
            log_odds = np.log((p / 1 - p) + 1e-16)
            for tree in self.forest:         # consecutively on each tree
                log_odds += self.lr * self.__reach_leaf(tree, x)            
            prob[row] = 1 / (1 + np.exp(log_odds))
        
        ret = prob if ret_prob is True else ((prob > 0.5) * 1).reshape(-1, 1)
        
        return ret
        
    def __reach_leaf(self, tree, x) -> float:
        """find which leaf would x end up and return its log(odds)"""
        node = tree
        while((node.left is not None) and (node.right is not None)):
            feature = node.data["feature"]
            split_value = node.data["split_value"]
            node = node.left if x[feature] <= split_value else node.right
        
        return node.data["log_odds"]
    
    def __get_partial_data(self, x, res, prob):
        # slice random samples and features, build a tree
        ROW_RATIO = 0.25
        COL_RATIO = 0.25
        rand_row_idx = random.sample(range(x.shape[0]), int(x.shape[0]*ROW_RATIO))
        rand_col_idx = random.sample(range(x.shape[1]), int(x.shape[1]*COL_RATIO))
        
        x_rand = x[rand_row_idx, rand_col_idx]
        res_rand = res[rand_row_idx]
        prob_rand = prob[rand_row_idx]

        return x_rand, res_rand, prob_rand

    def __grow_tree(self, x, res, prob, height=6):
        return self.__build_tree(x, res, prob, level=0, height=6)
        
    def __build_tree(self, x, res, prob, level, height):
        "build a tree by recursion"
        if level > height or x.shape[0] == 0: 
            return None
        else:
            weighted_quantiles = self.__get_weighted_quantiles(x, prob, quan_num=33)
            
            root = BTtreeNode()
            root.data = self.__find_best_split(x, res, prob, weighted_quantiles)
            
            if root.data["sim_gain"] < self.Gamma: 
                return None
            if level == height:
                log_odds = self.__merge_node(res, prob, Lambda=1)
                root.data["log_odds"] = log_odds
                return root
                
            x_left, res_left, prob_left, x_right, res_right, prob_right = self.__split_data(x, res, prob, root.data)
            
            root.left = self.__build_tree(x_left, res_left, prob_left, level+1, height)
            root.right = self.__build_tree(x_right, res_right, prob_right, level+1, height)
            
        return root

    def __split_data(self, x, res, prob, node_data):
        split_feature = node_data["feature"]
        split_value = node_data["split_value"]
        
        left_idx = np.where(x[split_feature] <= split_value)
        x_left, res_left, prob_left = x.iloc[left_idx, :], res[left_idx, :], prob[left_idx]
        
        right_idx = np.where(x[split_feature] > split_value)
        x_right, res_right, prob_right = x.iloc[right_idx, :], res[right_idx, :], prob[right_idx]
        
        return x_left, res_left, prob_left, x_right, res_right, prob_right 

    def __get_weighted_quantiles(self, x, prob, quan_num=33) -> np.array:
        """
        find split candidates by using weighted quantiles.
        Note that categorical features are treated as continuous here
        Even for a binary feature we have 33 split candidates
        Not sure if it's correct
        """
        quantiles = []
        for col in range(x.shape[1]):   # for each feature
            probx_col = np.concatenate([prob, x.iloc[:, col]])
            probx_col = probx_col[probx_col[:, 1].argsort()]    # sort by x, ascending

            prob_sorted = probx_col[:, 0]
            x_sorted = probx_col[:, 1:]
            
            weights = prob_sorted * (1 - prob_sorted)
            interval = np.sum(weights) / (quan_num + 1)

            cnt = 1
            weights_sum = 0
            quantiles_col = []
            for i in range(weights.shape[0]):
                weights_sum += weights[i]
                if weights_sum > interval * cnt:
                    quantiles_col.append(x_sorted[i])
                    cnt += 1
            quantiles.append(np.array([quantiles_col]).reshape(-1, 1))
            quantiles = np.array(quantiles)
            
        return quantiles
            
    def __find_best_split(self, x, res, prob, splits) -> dict:
        """Find the best split on current layer"""
        col_names = x.columns
        best_feature = ""
        best_split_value = 0
        sim_gain_max = 0
        for col in range(x.shape[1]):   # for each feature
            x_col = x.iloc[:, col]
            splits_col = splits[:, col]
            root_sim = self.__similarity_score(res, prob, Lambda=1)
            
            best_feature_local = 0
            best_split_value_local = 0
            sim_gain_max_local = 0
            for split in splits_col:    # for each split candidate in current feature
                left_res = res[np.where(x_col <= split)]
                left_prob = prob[np.where(x_col <= split)]
                right_res = res[np.where(x_col > split)]
                right_prob = prob[np.where(x_col > split)]
                
                left_node_sim = self.__similarity_score(left_res, left_prob, Lambda=1)
                right_node_sim = self.__similarity_score(right_res, right_prob, Lambda=1)
                
                sim_gain = left_node_sim + right_node_sim - root_sim
                
                if(sim_gain < 0): continue
                
                if(sim_gain > sim_gain_max_local):
                    sim_gain_max_local = sim_gain
                    best_feature_local = col_names[col]
                    best_split_value_local = split
            
            if(sim_gain_max_local > sim_gain_max):
                best_feature = best_feature_local
                best_split_value = best_split_value_local
                sim_gain_max = sim_gain_max_local
        
        node_data = {"feature": best_feature, "split_value": best_split_value, "sim_gain": sim_gain_max}
        
        return node_data

    def __similarity_score(self, res, prob, Lambda=1):
        return (np.sum(res) ** 2) / (np.sum(prob * (1-prob)) + Lambda)
    
    def __merge_node(self, res, prob, Lambda=1):
        """calculate log(odds) for a leaf node"""
        return np.sum(res) / ((prob * (1 - prob)) + Lambda)
    
    def summary(self):
        fig = plt.figure(figsize=(8, 3), dpi=100)

        plt.subplot(121)
        plt.plot(self.train_loss, color="orange", label="train loss")
        plt.plot(self.train_acc, color="royalblue", label="val loss")
        plt.legend()

        plt.subplot(122)
        plt.plot(self.val_loss, color="pink", label="train acc", )
        plt.plot(self.val_acc, color="purple", label="val acc")
        plt.legend()
        
        plt.show()