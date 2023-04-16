import numpy as np
import random
from DTtree import BTtreeNode

class XGBoostClassifier():
    def __init__(self,):
        pass
    
    def fit(self, x_train, y_train, x_val, y_val):
        
        EPOCH = 100 # tree number
        
        forest = []
        prob = np.zeros((x_train.shape[0], 1)) * 0.5
        
        for epoch in range(EPOCH):            
            x_rand, y_rand, prob_rand = self.get_partial_data(x_train, y_train, prob)
            tree, prob = self.grow_tree(x_rand, y_rand, prob_rand, height=6)
            forest.append(tree)
            
        # 
        # tree = self.grow_tree(weighted_quantiles, probx_train, y_train)
        # merge_leafs(tree)

    def get_partial_data(self, x, y, prob):
        # slice random samples and features, build a tree
        ROW_RATIO = 0.25
        COL_RATIO = 0.25
        rand_row_idx = random.sample(range(x.shape[0]), int(x.shape[0]*ROW_RATIO))
        rand_col_idx = random.sample(range(x.shape[1]), int(x.shape[1]*COL_RATIO))
        
        prob_rand = prob[rand_row_idx]
        x_rand = x[rand_row_idx, rand_col_idx]
        y_rand = y[rand_row_idx]

        return x_rand, y_rand, prob_rand

    def grow_tree(self, x, y, prob, height=6):
        return self.build_tree(x, y, prob, level=0, height=6)
        
    
    def build_tree(self, x, y, prob, level, height):
        "build a tree with recursion"
        if level > height: return
        else:
            weighted_quantiles = self.get_weighted_quantiles(x, prob, quan_num=33)
            root = self.find_best_split(x, y, prob, weighted_quantiles)
            
            

    def get_weighted_quantiles(self, x, prob, quan_num=33) -> np.array:
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
                    quantiles_col.append(x[i])
                    cnt += 1
            quantiles.append(np.array([quantiles_col]).reshape(-1, 1))
            quantiles = np.array(quantiles)
            
        return quantiles
            
    def find_best_split(self, x, y, prob, splits) -> dict:
        """Find the best split on current layer"""
        col_names = x.columns
        best_feature = ""
        best_split_value = 0
        sim_gain_max = 0
        for col in range(x.shape[1]):   # for each feature
            x_col = x.iloc[:, col]
            splits_col = splits[:, col]
            root_sim = self.similarity_score(x_col, prob, lbd=1)
            
            best_feature_local = 0
            best_split_value_local = 0
            sim_gain_max_local = 0
            for split in splits_col:    # for each split candidate in current feature
                left_node = x_col.iloc[np.where(x_col <= split)]
                left_prob = prob[np.where(x_col <= split)]
                right_node = x_col.iloc[np.where(x_col > split)]
                right_prob = prob[np.where(x_col > split)]
                
                left_node_sim = self.similarity_score(left_node, left_prob, lbd=1)
                right_node_sim = self.similarity_score(right_node, right_prob, lbd=1)
                
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
        
        node = {"feature": best_feature, "split_value": best_split_value, "sim_gain": sim_gain_max}
        
        return node

    def similarity_score(self, x, prob, lbd=1):
        return (np.sum(x) ** 2) / (np.sum(prob * (1-prob)) + lbd)
    
    def merge_logodds(self, leaf, lbd=1):
        prob = leaf[:, 0]
        res = leaf[:, 1:]
        return np.sum(res) / ((prob * (1 - prob)) + lbd)
    