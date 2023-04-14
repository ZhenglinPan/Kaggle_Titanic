import numpy as np
import random

class XGBoostClassifier():
    def __init__(self,):
        pass
    
    def fit(self, x_train, y_train, x_val, y_val):
        prob = np.zeros((x_train.shape[0], 1)) * 0.5
        probx_train = np.concatenate([prob, x_train], axis=1)
        weighted_quantiles = self.get_weighted_quantiles(probx_train, quan_num=33)
        tree = self.grow_tree(weighted_quantiles, probx_train, y_train)
        merge_leafs(tree)


    def get_weighted_quantiles(self, probx, quan_num=33) -> np.array:
        prob = probx[:, 0]
        quantiles = []
        for col in range(1, probx.shape[1]):
            probx_col = np.concatenate([prob, probx[:, col]])
            probx_col = probx_col[probx_col[:, 1].argsort()]

            prob_c = probx_col[:, 0]
            x = probx_col[:, 1:]
            
            weights = prob_c * (1 - prob_c)
            weights_sum = np.sum(weights)
            interval = weights_sum / (quan_num + 1)

            cnt = 1
            w_sum = 0
            quantiles_col = []
            for i in range(weights.shape[0]):
                w_sum += weights[i]
                if w_sum > interval * cnt:
                    quantiles_col.append(x[i])
                    cnt += 1

            quantiles.append(np.array([quantiles_col]).reshape(-1, 1))
            
            return quantiles
        
        
    def grow_tree(self, splits, probx, y):
        # slice random samples and features, build a tree
        COL_RATIO = 0.25
        ROW_RATIO = 0.25
        rand_col_idx = random.sample(range(1, probx.shape[1]), int(probx.shape[1]*COL_RATIO))
        rand_row_idx = random.sample(range(probx.shape[0]), int(probx.shape[0]*ROW_RATIO))
        
        prob = probx[rand_row_idx, 0]
        x = probx[rand_row_idx, rand_col_idx]
        y = y[rand_row_idx]
        splits = splits[rand_row_idx, rand_col_idx]
        
        tree = self.build_DTtree(prob, x, y, splits)
        
        return tree

    
    def build_DTtree(self, prob, x, y, quantiles, height=6):
        tree = []
        tree.append(x)
        for h in range(height):
            best_col = 0
            best_split = 0
            sim_gain_max = 0
            for col in range(x.shape[1]):
                x_col = x[:, col]
                quantile_col = quantiles[:, col]
                root_sim = self.similarity_score(x_col, prob, lbd=1)
                
                best_quantile_local = 0
                sim_gain_max_local = 0
                for quantile in quantile_col:
                    left_node = x_col[np.where(x_col <= quantile)]
                    left_prob = prob[np.where(x_col <= quantile)]
                    right_node = x_col[np.where(x_col > quantile)]
                    right_prob = prob[np.where(x_col > quantile)]
                    
                    left_node_sim = self.similarity_score(left_node, left_prob, lbd=1)
                    right_node_sim = self.similarity_score(right_node, right_prob, lbd=1)
                    
                    sim_gain = left_node_sim + right_node_sim - root_sim
                    
                    if(sim_gain < 0): continue
                    
                    if(sim_gain > sim_gain_max_local):
                        sim_gain_max_local = sim_gain
                        best_quantile_local = quantile
                
                if(sim_gain_max_local > sim_gain_max):
                    sim_gain_max = sim_gain_max_local
                    best_split = best_quantile_local
                    best_col = col
                    
            # by far we've found the best feature for current layer
            
            

    def similarity_score(self, node, prob, lbd=1):
        return (np.sum(node) ** 2) / (np.sum(prob * (1-prob)) + lbd)
    
    def merge_logodds(self, leaf, lbd=1):
        prob = leaf[:, 0]
        res = leaf[:, 1:]
        return np.sum(res) / ((prob * (1 - prob)) + lbd)
    