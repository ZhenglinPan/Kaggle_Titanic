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
        self.cover_threshold = config["cover"]
        self.lr = config["learning_rate"]
        self.quan_num = config["quantiles"]
        self.height = config["tree_height"]
        self.feature_ratio = config["feature_ratio"]
        self.sample_ratio = config["sample_ratio"]
    
    def fit(self, x_train, y_train, x_val, y_val):
        
        y_train = np.array(y_train).reshape(-1, 1)
        y_val = np.array(y_val).reshape(-1, 1)
        
        prob_train = np.ones((x_train.shape[0], 1)) * 0.5
        res_train = y_train - prob_train
        
        for epoch in tqdm(range(self.EPOCH)):            
            x_rand, res_rand, prob_rand = self.__get_partial_data(x_train, res_train, prob_train)
            
            tree = self.__grow_tree(x_rand, res_rand, prob_rand)
            if tree is None:
                print(epoch, "Tree is pruned to None.")
                continue
            
            self.forest.append(tree)
            
            prob_train = self.__update(tree, x_train, prob_train)
            res_train = y_train - prob_train
            
            pred_train = ((prob_train > 0.5) * 1).reshape(-1, 1)
            self.train_loss.append(self.__cross_entropy_loss(y_train, prob_train))
            self.train_acc.append(np.mean(y_train == pred_train))
            
            prob_val = self.predict(x_val, ret_prob=True)
            pred_val = self.predict(x_val, ret_prob=False)
            self.val_loss.append(self.__cross_entropy_loss(y_val, prob_val))
            self.val_acc.append(np.mean(y_val == pred_val))

    def predict(self, X, ret_prob=False):
        """Predict the probability of X on forest with initial prob"""
        prob = np.ones((X.shape[0], 1)) * 0.5
        for row in range(X.shape[0]):   # for each input x
            x = X.iloc[row, :]
            p = prob[row]
            log_odds = np.log((p / (1 - p)) + 1e-16)
            for tree in self.forest:         # consecutively on each tree
                log_odds += self.lr * self.__reach_leaf(tree, x)            
            prob[row] = np.exp(log_odds) / (1 + np.exp(log_odds))
        
        ret = prob if ret_prob is True else ((prob > 0.5) * 1).reshape(-1, 1)
        
        return ret

    def __get_weighted_quantiles(self, x, prob) -> list:
        """
        Find split candidates by using weighted quantiles.
        Quantiles number equals min(self.quan_num, x.shape[0]).
        Treat categorical variables as numerical data is OK,
        but would take extra computation.
        """
        quantiles = []
        for col in range(x.shape[1]):   # for each feature
            if(len(np.unique(x)) > self.quan_num):
                x_col = np.array(x.iloc[:, col]).reshape(-1, 1)
                probx_col = np.hstack([prob, x_col])
                probx_col = probx_col[probx_col[:, 1].argsort()]    # sort by x, ascending

                prob_sorted = probx_col[:, 0]
                x_sorted = probx_col[:, 1]
                
                weights = prob_sorted * (1 - prob_sorted)
                interval = np.sum(weights) / (self.quan_num + 1)

                if interval == 0:
                    raise ValueError("Interval must be greater than zero.")
                
                cnt = 0
                weights_sum = 0
                quantiles_col = []
                for i in range(weights.shape[0]):
                    weights_sum += weights[i]
                    if weights_sum > interval * (cnt + 1) + 1e-5:
                        quantiles_col.append(x_sorted[i])
                        cnt += 1
            else:
                quantiles_col = list(np.unique(x))
            quantiles.append(quantiles_col)     # quantiles size (63, )
        
        return quantiles

    def __update(self, tree, X, pre_prob):
        """Update the prob for X on a tree given previous prob"""
        prob = pre_prob.copy()
        for row in range(X.shape[0]):   # for each input x
            x = X.iloc[row, :]
            p = prob[row]
            log_odds = np.log((p / (1 - p)) + 1e-16) + self.lr * self.__reach_leaf(tree, x)
            prob[row] = np.exp(log_odds) / (1 + np.exp(log_odds))
        return prob
        
    def __reach_leaf(self, tree, x) -> float:
        """find which leaf would x end up and return its log(odds)"""
        node = tree
        while((node.left is not None) and (node.right is not None)):    # might be problematic if there exists nodes with only one leaf
            feature = node.data["feature"]
            split_value = node.data["split_value"]
            node = node.left if x[feature] <= split_value else node.right
        
        return node.data["log_odds"]
    
    def __get_partial_data(self, x, res, prob):
        # slice random samples and features, build a tree
        rand_row_idx = random.sample(range(x.shape[0]), int(x.shape[0]*self.sample_ratio))
        rand_col_idx = random.sample(range(x.shape[1]), int(x.shape[1]*self.feature_ratio))
        
        x_rand = x.iloc[rand_row_idx, rand_col_idx]
        res_rand = res[rand_row_idx]
        prob_rand = prob[rand_row_idx]

        return x_rand, res_rand, prob_rand

    def __grow_tree(self, x, res, prob):
        return self.__build_tree(x, res, prob, level=0)
        
    def __build_tree(self, x, res, prob, level=0):
        "build a tree by recursion"
        if level > self.height or x.shape[0] == 0: 
            return None
        else:
            weighted_quantiles = self.__get_weighted_quantiles(x, prob)
            
            root = BTtreeNode()
            root.data = self.__find_best_split(x, res, prob, weighted_quantiles)
            
            con1 = (root.data != -1)  # on current x, not splitting is preferred, sim_gain is otherwise < 1
            con2 = (self.__cover_score(prob) > self.cover_threshold)  # node is otherwise too small
            if con1 and con2:
                con3 = (root.data["sim_gain"] > self.Gamma)       # sim gain is otherwise too small
                if con3:
                    x_left, res_left, prob_left, x_right, res_right, prob_right = self.__split_data(x, res, prob, root.data)
                    
                    root.left = self.__build_tree(x_left, res_left, prob_left, level+1)
                    root.right = self.__build_tree(x_right, res_right, prob_right, level+1)
            else:
                root.data = dict()  # else take current x as a leaf node
                root.data["sim_gain"] = self.__similarity_score(res, prob)
                
            if (root.left is None) and (root.right is None):
                log_odds = self.__merge_node(res, prob)
                root.data["log_odds"] = log_odds
            
        return root
    
    def __split_data(self, x, res, prob, node_data):
        split_feature = node_data["feature"]
        split_value = node_data["split_value"]
        
        left_idx = np.where(x[split_feature] <= split_value)
        x_left, res_left, prob_left = x.iloc[left_idx], res[left_idx], prob[left_idx]

        right_idx = np.where(x[split_feature] > split_value)
        x_right, res_right, prob_right = x.iloc[right_idx], res[right_idx], prob[right_idx]
        
        return x_left, res_left, prob_left, x_right, res_right, prob_right 
            
    def __find_best_split(self, x, res, prob, splits) -> dict:
        """
        Find the best split on random part of x for current layer
        return -1 when sim_gain < 0, indicating not spliting is better.
        """
        col_names = x.columns
        best_feature = ""
        best_split_value = 0
        sim_gain_max = 0
        for col in range(x.shape[1]):   # for each feature
            x_col = np.array(x.iloc[:, col]).reshape(-1, 1)
            splits_col = splits[col]
            root_sim = self.__similarity_score(res, prob)
            
            best_feature_local = 0
            best_split_value_local = 0
            sim_gain_max_local = 0
            for split in splits_col:    # for each split candidate in current feature
                left_res = res[np.where(x_col <= split)]
                left_prob = prob[np.where(x_col <= split)]
                right_res = res[np.where(x_col > split)]
                right_prob = prob[np.where(x_col > split)]
                
                left_node_sim = self.__similarity_score(left_res, left_prob)
                right_node_sim = self.__similarity_score(right_res, right_prob)
                
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
        
        if best_feature == "" and best_split_value == 0 and sim_gain_max == 0:
            node_data = -1
        else:
            node_data = {"feature": best_feature, 
                         "split_value": best_split_value, 
                         "sim_gain": sim_gain_max,
                         }
            
        return node_data

    def __similarity_score(self, res, prob):
        return (np.sum(res) ** 2) / (np.sum(prob * (1-prob)) + self.Lambda)
    
    def __merge_node(self, res, prob):
        """calculate log(odds) for a leaf node"""
        return np.sum(res) / (np.sum(prob * (1 - prob)) + self.Lambda)

    def __cross_entropy_loss(self, truth, pred):
        return (-1 / truth.shape[1]) * np.sum(truth * np.log(pred + 1e-16))
    
    def __cover_score(self, prob):
        return np.sum(prob * (1 - prob))
    
    def summary(self):
        fig = plt.figure(figsize=(8, 3), dpi=100)

        plt.subplot(121)
        plt.xlabel("EPOCH")
        plt.ylabel("CE loss")
        plt.plot(self.train_loss, color="orange", label="train loss on a tree")
        plt.plot(self.val_loss, color="royalblue", label="val loss on the forest")
        plt.legend()

        plt.subplot(122)
        plt.xlabel("EPOCH")
        plt.ylabel("Accuracy")
        plt.plot(self.train_acc, color="pink", label="train acc on a tree")
        plt.plot(self.val_acc, color="purple", label="val acc on the forest")
        plt.legend()
        
        plt.savefig('summary.png', bbox_inches="tight")
        plt.show()