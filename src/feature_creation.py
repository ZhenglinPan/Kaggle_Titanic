import numpy as np
import pandas as pd

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

def Binary_Cross_Combination(colNames, features, OneHot=True):
    """
    分类变量两两组合交叉衍生函数
    
    :param colNames: 参与交叉衍生的列名称
    :param features: 原始数据集
    :param OneHot: 是否进行独热编码
    
    :return: 交叉衍生后的新特征和新列名称
    """
    
    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []
    
    # 提取需要进行交叉组合的特征
    features = features[colNames]
    
    # 逐个创造新特征名称、新特征
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            
            newNames = col_name + '&' + colNames[col_sub_index]
            colNames_new_l.append(newNames)
            
            newDF = pd.Series(features[col_name].astype('str')  
                              + '&'
                              + features[colNames[col_sub_index]].astype('str'), 
                              name=col_name)
            features_new_l.append(newDF)
    
    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l
    
    # 对新特征矩阵进行独热编码
    if OneHot == True:
        enc = preprocessing.OneHotEncoder()
        enc.fit_transform(features_new)
        colNames_new = cate_colName(enc, colNames_new_l, drop=None)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=colNames_new)
        
    return features_new, colNames_new



def Binary_PolynomialFeatures(colNames, degree, features):
    # from sklearn.preprocessing import PolynomialFeatures
    """
    连续变量两变量多项式衍生函数
    
    :param colNames: 参与交叉衍生的列名称
    :param degree: 多项式最高阶
    :param features: 原始数据集
    
    :return: 交叉衍生后的新特征和新列名称
    """
    
    # 创建空列表存储器
    colNames_new_l = []
    features_new_l = []
    
    # 提取需要进行多项式衍生的特征
    features = features[colNames]
    
    # 逐个进行多项式特征组合
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            col_temp = [col_name] + [colNames[col_sub_index]]
            array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features[col_temp])
            features_new_l.append(pd.DataFrame(array_new_temp[:, 2:]))
    
            # 逐个创建衍生多项式特征的名称
            for deg in range(2, degree+1):
                for i in range(deg+1):
                    col_name_temp = col_temp[0] + '**' + str(deg-i) + '*'+ col_temp[1] + '**' + str(i)
                    colNames_new_l.append(col_name_temp)
           
    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l
    
    return features_new, colNames_new


def Binary_Group_Statistics(keyCol, 
                            features, 
                            col_num=None, 
                            col_cat=None, 
                            num_stat=['mean', 'var', 'max', 'min', 'skew', 'median'], 
                            cat_stat=['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'], 
                            quant=True):
    """
    双变量分组统计特征衍生函数
    
    :param keyCol: 分组参考的关键变量
    :param features: 原始数据集
    :param col_num: 参与衍生的连续型变量
    :param col_cat: 参与衍生的离散型变量
    :param num_stat: 连续变量分组统计量
    :param cat_num: 离散变量分组统计量  
    :param quant: 是否计算分位数  

    :return: 交叉衍生后的新特征和新特征的名称
    """
    
    # 当输入的特征有连续型特征时
    if col_num != None:
        aggs_num = {}
        colNames = col_num
        
        # 创建agg方法所需字典
        for col in col_num:
            aggs_num[col] = num_stat 
            
        # 创建衍生特征名称列表
        cols_num = [keyCol]
        for key in aggs_num.keys():
            cols_num.extend([key+'_'+keyCol+'_'+stat for stat in aggs_num[key]])
            
        # 创建衍生特征df
        features_num_new = features[col_num+[keyCol]].groupby(keyCol).agg(aggs_num).reset_index()
        features_num_new.columns = cols_num 
        
        # 当输入的特征有连续型也有离散型特征时
        if col_cat != None:        
            aggs_cat = {}
            colNames = col_num + col_cat

            # 创建agg方法所需字典
            for col in col_cat:
                aggs_cat[col] = cat_stat

            # 创建衍生特征名称列表
            cols_cat = [keyCol]
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+keyCol+'_'+stat for stat in aggs_cat[key]])    

            # 创建衍生特征df
            features_cat_new = features[col_cat+[keyCol]].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat
    
            # 合并连续变量衍生结果与离散变量衍生结果
            df_temp = pd.merge(features_num_new, features_cat_new, how='left',on=keyCol)
            features_new = pd.merge(features[keyCol], df_temp, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_num + cols_cat
            colNames_new.remove(keyCol)
            colNames_new.remove(keyCol)
         
        # 当只有连续变量时
        else:
            # merge连续变量衍生结果与原始数据，然后删除重复列
            features_new = pd.merge(features[keyCol], features_num_new, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_num
            colNames_new.remove(keyCol)
    
    # 当没有输入连续变量时
    else:
        # 但存在分类变量时，即只有分类变量时
        if col_cat != None:
            aggs_cat = {}
            colNames = col_cat

            for col in col_cat:
                aggs_cat[col] = cat_stat

            cols_cat = [keyCol]
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+keyCol+'_'+stat for stat in aggs_cat[key]])    

            features_cat_new = features[col_cat+[keyCol]].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat            
             
            features_new = pd.merge(features[keyCol], features_cat_new, how='left',on=keyCol)
            features_new.loc[:, ~features_new.columns.duplicated()]
            colNames_new = cols_cat
            colNames_new.remove(keyCol) 
    
    if quant:
        # 定义四分位计算函数
        def q1(x):
            """
            下四分位数
            """
            return x.quantile(0.25)

        def q2(x):
            """
            上四分位数
            """
            return x.quantile(0.75)

        aggs = {}
        for col in colNames:
            aggs[col] = ['q1', 'q2']

        cols = [keyCol]
        for key in aggs.keys():
            cols.extend([key+'_'+keyCol+'_'+stat for stat in aggs[key]])    

        aggs = {}
        for col in colNames:
            aggs[col] = [q1, q2]    

        features_temp = features[colNames+[keyCol]].groupby(keyCol).agg(aggs).reset_index()
        features_temp.columns = cols

        features_new = pd.merge(features_new, features_temp, how='left',on=keyCol)
        features_new.loc[:, ~features_new.columns.duplicated()]
        colNames_new = colNames_new + cols
        colNames_new.remove(keyCol)     
    
    features_new.drop([keyCol], axis=1, inplace=True)
        
    return features_new, colNames_new


def Target_Encode(keyCol, 
                  X_train, 
                  y_train,
                  X_test, 
                  col_num=None, 
                  col_cat=None, 
                  num_stat=['mean', 'var', 'max', 'min', 'skew', 'median'], 
                  cat_stat=['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'], 
                  quant=True, 
                  multi=False, 
                  extension=False,
                  n_splits=5, 
                  random_state=42):
    """
    目标编码
    
    :param keyCol: 分组参考的关键变量
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征
    :param col_num: 参与衍生的连续型变量
    :param col_cat: 参与衍生的离散型变量
    :param num_stat: 连续变量分组统计量
    :param cat_num: 离散变量分组统计量  
    :param quant: 是否计算分位数  
    :param multi: 是否进行多变量的分组统计特征衍生
    :param extension: 是否进行二阶特征衍生
    :param n_splits: 进行几折交叉统计  
    :param random_state: 随机数种子  

    :return: 目标编码后的新特征和新特征的名称
    """
        
    # 获取标签名称
    target = y_train.name
    # 合并同时带有特征和标签的完整训练集
    train = pd.concat([X_train, y_train], axis=1)
    
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 每一折验证集的结果存储容器
    df_l = []
    
    # 进行交叉统计
    for trn_idx, val_idx in folds.split(train):
        trn_temp = train.iloc[trn_idx]
        val_temp = train.iloc[val_idx]
        trn_temp_new, val_temp_new, colNames_trn_temp_new, colNames_val_temp_new = Group_Statistics(keyCol, 
                                                                                                    X_train = trn_temp, 
                                                                                                    X_test = val_temp, 
                                                                                                    col_num = col_num, 
                                                                                                    col_cat = col_cat, 
                                                                                                    num_stat = num_stat, 
                                                                                                    cat_stat = cat_stat, 
                                                                                                    quant = quant, 
                                                                                                    multi = multi, 
                                                                                                    extension = extension)
        val_temp_new.index = val_temp.index
        df_l.append(val_temp_new)
    
    # 创建训练集的衍生特征
    features_train_new = pd.concat(df_l).sort_index(ascending=True) 
    colNames_train_new = [col + '_kfold' for col in features_train_new.columns]
    features_train_new.columns = colNames_train_new
    
    # 对测试集结果进行填补
    features_train_new, features_test_new, colNames_train_new, colNames_test_new = test_features(keyCol = keyCol, 
                                                                                                 X_train = X_train, 
                                                                                                 X_test = X_test, 
                                                                                                 features_train_new = features_train_new,
                                                                                                 multi = multi)
   
    # 如果特征不一致，则进行0值填补
    if colNames_train_new != colNames_test_new:
        features_train_new, features_test_new, colNames_train_new, colNames_test_new = Features_Padding(features_train_new = features_train_new, 
                                                                                                        features_test_new = features_test_new, 
                                                                                                        colNames_train_new = colNames_train_new, 
                                                                                                        colNames_test_new = colNames_test_new)
        
    assert colNames_train_new  == colNames_test_new
    return features_train_new, features_test_new, colNames_train_new, colNames_test_new


def Group_Statistics(keyCol, 
                     X_train, 
                     X_test, 
                     col_num=None, 
                     col_cat=None, 
                     num_stat=['mean', 'var', 'max', 'min', 'skew', 'median'], 
                     cat_stat=['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'], 
                     quant=True, 
                     multi=False, 
                     extension=False):
    """
    分组统计特征衍生函数
    
    :param keyCol: 分组参考的关键变量
    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param col_num: 参与衍生的连续型变量
    :param col_cat: 参与衍生的离散型变量
    :param num_stat: 连续变量分组统计量
    :param cat_num: 离散变量分组统计量  
    :param quant: 是否计算分位数  
    :param multi: 是否进行多变量的分组统计特征衍生
    :param extension: 是否进行二阶特征衍生

    :return: 分组统计衍生后的新特征和新特征的名称
    """
    
    # 进行训练集的特征衍生
    if multi == False:
        # 进行双变量的交叉衍生
        features_train_new, colNames_train_new = Binary_Group_Statistics(keyCol = keyCol, 
                                                                         features = X_train, 
                                                                         col_num = col_num, 
                                                                         col_cat = col_cat, 
                                                                         num_stat = num_stat, 
                                                                         cat_stat = cat_stat, 
                                                                         quant = quant)
        # 是否进一步进行二阶特征衍生
        if extension == True:
            if col_num == None:
                colNames = col_cat
            elif col_cat == None:
                colNames = col_num
            else:
                colNames = col_num + col_cat
                
            features_train_new_ex, colNames_train_new_ex = Group_Statistics_Extension(colNames = colNames,
                                                                                      keyCol = keyCol,
                                                                                      features = X_train)
            
            features_train_new = pd.concat([features_train_new, features_train_new_ex], axis=1)
            colNames_train_new.extend(colNames_train_new_ex)
            
        
    else:
        # 进行多变量的交叉衍生
        features_train_new, colNames_train_new = Multi_Group_Statistics(keyCol = keyCol, 
                                                                        features = X_train, 
                                                                        col_num = col_num, 
                                                                        col_cat = col_cat, 
                                                                        num_stat = num_stat, 
                                                                        cat_stat = cat_stat, 
                                                                        quant = quant)

    
    # 对测试集结果进行填补
    features_train_new, features_test_new, colNames_train_new, colNames_test_new = test_features(keyCol, 
                                                                                                 X_train, 
                                                                                                 X_test, 
                                                                                                 features_train_new,
                                                                                                 multi=multi)
    # 如果特征不一致，则进行0值填补
    # 对于分组统计特征来说一般不会出现该情况
    if colNames_train_new != colNames_test_new:
        features_train_new, features_test_new, colNames_train_new, colNames_test_new = Features_Padding(features_train_new = features_train_new, 
                                                                                                        features_test_new = features_test_new, 
                                                                                                        colNames_train_new = colNames_train_new, 
                                                                                                        colNames_test_new = colNames_test_new)
        
    assert colNames_train_new  == colNames_test_new
    return features_train_new, features_test_new, colNames_train_new, colNames_test_new


def Group_Statistics_Extension(colNames, keyCol, features):
    """
    双变量分组统计二阶特征衍生函数
    
    :param colNames: 参与衍生的特征
    :param keyCol: 分组参考的关键变量
    :param features: 原始数据集
    
    :return: 交叉衍生后的新特征和新列名称
    """
    
    # 定义四分位计算函数
    def q1(x):
        """
        下四分位数
        """
        return x.quantile(0.25)

    def q2(x):
        """
        上四分位数
        """
        return x.quantile(0.75)   
    
    # 一阶特征衍生
    # 先定义用于生成列名称的aggs
    aggs = {}    
    for col in colNames:
        aggs[col] = ['mean', 'var', 'median', 'q1', 'q2']       
    cols = [keyCol]
    for key in aggs.keys():
        cols.extend([key+'_'+keyCol+'_'+stat for stat in aggs[key]])

    # 再定义用于进行分组汇总的aggs
    aggs = {}   
    for col in colNames:
        aggs[col] = ['mean', 'var', 'median', q1, q2] 
           
    features_new = features[colNames+[keyCol]].groupby(keyCol).agg(aggs).reset_index()
    features_new.columns = cols
             
    features_new = pd.merge(features[keyCol], features_new, how='left',on=keyCol)
    features_new.loc[:, ~features_new.columns.duplicated()]
    colNames_new = cols
    colNames_new.remove(keyCol)
    
    # 二阶特征衍生
    # 流量平滑特征
    for col_temp in colNames:
        col = col_temp+'_'+keyCol+'_'+'mean'
        features_new[col_temp+'_dive1_'+col] = features_new[keyCol] / (features_new[col] + 1e-5)
        colNames_new.append(col_temp+'_dive1_'+col)
        col = col_temp+'_'+keyCol+'_'+'median'
        features_new[col_temp+'_dive2_'+col] = features_new[keyCol] / (features_new[col] + 1e-5)
        colNames_new.append(col_temp+'_dive2_'+col)
        
    # 黄金组合特征
    for col_temp in colNames:
        col = col_temp+'_'+keyCol+'_'+'mean'
        features_new[col_temp+'_minus1_'+col] = features_new[keyCol] - features_new[col] 
        colNames_new.append(col_temp+'_minus1_'+col)
        col = col_temp+'_'+keyCol+'_'+'median'
        features_new[col_temp+'_minus2_'+col] = features_new[keyCol] - features_new[col] 
        colNames_new.append(col_temp+'_minus2_'+col)
        
    # 组内归一化特征
    for col_temp in colNames:
        col_mean = col_temp+'_'+keyCol+'_'+'mean'
        col_var = col_temp+'_'+keyCol+'_'+'var'
        features_new[col_temp+'_norm_'+keyCol] = (features_new[keyCol] - features_new[col_mean]) / (np.sqrt(features_new[col_var]) + 1e-5)      
        colNames_new.append(col_temp+'_norm_'+keyCol)
    
    # Gap特征
    for col_temp in colNames:
        col_q1 = col_temp+'_'+keyCol+'_'+'q1'
        col_q2 = col_temp+'_'+keyCol+'_'+'q2'
        features_new[col_temp+'_gap_'+keyCol] = features_new[col_q2] - features_new[col_q1]  
        colNames_new.append(col_temp+'_gap_'+keyCol)
        
    # 数据倾斜特征
    for col_temp in colNames:
        col_mean = col_temp+'_'+keyCol+'_'+'mean'
        col_median = col_temp+'_'+keyCol+'_'+'median'
        features_new[col_temp+'_mag1_'+keyCol] = features_new[col_median] - features_new[col_mean]    
        colNames_new.append(col_temp+'_mag1_'+keyCol)
        features_new[col_temp+'_mag2_'+keyCol] = features_new[col_median] / (features_new[col_mean] + 1e-5)
        colNames_new.append(col_temp+'_mag2_'+keyCol)
        
    # 变异系数
    for col_temp in colNames:
        col_mean = col_temp+'_'+keyCol+'_'+'mean'
        col_var = col_temp+'_'+keyCol+'_'+'var'
        features_new[col_temp+'_cv_'+keyCol] = np.sqrt(features_new[col_var]) / (features_new[col_mean] + 1e-5)
        colNames_new.append(col_temp+'_cv_'+keyCol)

    features_new.drop([keyCol], axis=1, inplace=True)
    return features_new, colNames_new



def Multi_Group_Statistics(keyCol, 
                           features, 
                           col_num=None, 
                           col_cat=None, 
                           num_stat=['mean', 'var', 'max', 'min', 'skew', 'median'], 
                           cat_stat=['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'], 
                           quant=True):
    """
    多变量分组统计特征衍生函数
    
    :param keyCol: 分组参考的关键变量
    :param features: 原始数据集
    :param col_num: 参与衍生的连续型变量
    :param col_cat: 参与衍生的离散型变量
    :param num_stat: 连续变量分组统计量
    :param cat_num: 离散变量分组统计量  
    :param quant: 是否计算分位数  

    :return: 交叉衍生后的新特征和新特征的名称
    """
    # 生成原数据合并的主键
    features_key1, col1 = Multi_Cross_Combination(keyCol, features, OneHot=False)
    
    # 当输入的特征有连续型特征时
    if col_num != None:
        aggs_num = {}
        colNames = col_num
        
        # 创建agg方法所需字典
        for col in col_num:
            aggs_num[col] = num_stat 
            
        # 创建衍生特征名称列表
        cols_num = keyCol.copy()

        for key in aggs_num.keys():
            cols_num.extend([key+'_'+col1+'_'+stat for stat in aggs_num[key]])
            
        # 创建衍生特征df
        features_num_new = features[col_num+keyCol].groupby(keyCol).agg(aggs_num).reset_index()
        features_num_new.columns = cols_num 
        
        # 生成主键
        features_key2, col2 = Multi_Cross_Combination(keyCol, features_num_new, OneHot=False)
        
        # 创建包含合并主键的数据集
        features_num_new = pd.concat([features_key2, features_num_new], axis=1)
        
        
        # 当输入的特征有连续型也有离散型特征时
        if col_cat != None:        
            aggs_cat = {}
            colNames = col_num + col_cat

            # 创建agg方法所需字典
            for col in col_cat:
                aggs_cat[col] = cat_stat

            # 创建衍生特征名称列表
            cols_cat = keyCol.copy()
            
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+col1+'_'+stat for stat in aggs_cat[key]])    

            # 创建衍生特征df
            features_cat_new = features[col_cat+keyCol].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat
            
            # 生成主键
            features_key3, col3 = Multi_Cross_Combination(keyCol, features_cat_new, OneHot=False)

            # 创建包含合并主键的数据集
            features_cat_new = pd.concat([features_key3, features_cat_new], axis=1)            
    
    
            # 合并连续变量衍生结果与离散变量衍生结果
            # 合并新的特征矩阵
            df_temp = pd.concat([features_num_new, features_cat_new], axis=1)
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            # 将新的特征矩阵与原始数据集合并
            features_new = pd.merge(features_key1, df_temp, how='left',on=col1)
         
        
        # 当只有连续变量时
        else:
            # merge连续变量衍生结果与原始数据，然后删除重复列
            features_new = pd.merge(features_key1, features_num_new, how='left',on=col1)
            features_new = features_new.loc[:, ~features_new.columns.duplicated()]
    
    # 当没有输入连续变量时
    else:
        # 但存在分类变量时，即只有分类变量时
        if col_cat != None:
            aggs_cat = {}
            colNames = col_cat

            for col in col_cat:
                aggs_cat[col] = cat_stat

            cols_cat = keyCol.copy()
            
            for key in aggs_cat.keys():
                cols_cat.extend([key+'_'+col1+'_'+stat for stat in aggs_cat[key]])    

            features_cat_new = features[col_cat+keyCol].groupby(keyCol).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat            
             
            features_new = pd.merge(features_key1, features_cat_new, how='left',on=col1)
            features_new = features_new.loc[:, ~features_new.columns.duplicated()]
    
    if quant:
        # 定义四分位计算函数
        def q1(x):
            """
            下四分位数
            """
            return x.quantile(0.25)

        def q2(x):
            """
            上四分位数
            """
            return x.quantile(0.75)

        aggs = {}
        for col in colNames:
            aggs[col] = ['q1', 'q2']

        cols = keyCol.copy()
        
        for key in aggs.keys():
            cols.extend([key+'_'+col1+'_'+stat for stat in aggs[key]])    

        aggs = {}
        for col in colNames:
            aggs[col] = [q1, q2]    

        features_temp = features[colNames+keyCol].groupby(keyCol).agg(aggs).reset_index()
        features_temp.columns = cols
        features_new.drop(keyCol, axis=1, inplace=True)
    
        # 生成主键
        features_key4, col4 = Multi_Cross_Combination(keyCol, features_temp, OneHot=False)
        
        # 创建包含合并主键的数据集
        features_temp = pd.concat([features_key4, features_temp], axis=1)        

        # 合并新特征矩阵
        features_new = pd.merge(features_new, features_temp, how='left',on=col1)
        features_new = features_new.loc[:, ~features_new.columns.duplicated()]
  

    features_new.drop(keyCol+[col1], axis=1, inplace=True)
    colNames_new = list(features_new.columns)
    
    return features_new, colNames_new



def test_features(keyCol,
                  X_train, 
                  X_test,
                  features_train_new,
                  multi=False):
    """
    测试集特征填补函数
    
    :param keyCol: 分组参考的关键变量
    :param X_train: 训练集特征
    :param X_test: 测试集特征
    :param features_train_new: 训练集衍生特征
    :param multi: 是否多变量参与分组
    
    :return: 分组统计衍生后的新特征和新特征的名称
    """
    
    # 创建主键
    # 创建带有主键的训练集衍生特征df
    # 创建只包含主键的test_key
    if multi == False:
        keyCol = keyCol
        features_train_new[keyCol] = X_train[keyCol].reset_index()[keyCol]
        test_key = pd.DataFrame(X_test[keyCol])
    else:
        train_key, train_col = Multi_Cross_Combination(colNames=keyCol, features=X_train, OneHot=False)
        test_key, test_col = Multi_Cross_Combination(colNames=keyCol, features=X_test, OneHot=False)
        assert train_col == test_col
        keyCol = train_col
        features_train_new[keyCol] = train_key[train_col].reset_index()[train_col]
        
    # 利用groupby进行去重
    features_test_or = features_train_new.groupby(keyCol).mean().reset_index()
    
    # 和测试集进行拼接
    features_test_new = pd.merge(test_key, features_test_or, on=keyCol, how='left')
    
    # 删除keyCol列，只保留新衍生的列
    features_test_new.drop([keyCol], axis=1, inplace=True)
    features_train_new.drop([keyCol], axis=1, inplace=True)
    
    # 输出列名称
    colNames_train_new = list(features_train_new.columns)
    colNames_test_new = list(features_test_new.columns)
    
    return features_train_new, features_test_new, colNames_train_new, colNames_test_new



def Features_Padding(features_train_new, 
                     features_test_new, 
                     colNames_train_new, 
                     colNames_test_new):
    """
    特征零值填补函数
    
    :param features_train_new: 训练集衍生特征
    :param features_test_new: 测试集衍生特征
    :param colNames_train_new: 训练集衍生列名称
    :param colNames_test_new: 测试集衍生列名称
    
    :return: 0值填补后的新特征和特征名称
    """
    if len(colNames_train_new) > len(colNames_test_new):
        sub_colNames = list(set(colNames_train_new) - set(colNames_test_new))
        
        for col in sub_colNames:
            features_test_new[col] = 0
        
        features_test_new = features_test_new[colNames_train_new]
        colNames_test_new = list(features_test_new.columns)
            
    elif len(colNames_train_new) < len(colNames_test_new):
        sub_colNames = list(set(colNames_test_new) - set(colNames_train_new))
        
        for col in sub_colNames:
            features_train_new[col] = 0
        
        features_train_new = features_train_new[colNames_test_new]
        colNames_train_new = list(features_train_new.columns)    
    assert colNames_train_new  == colNames_test_new
    return features_train_new, features_test_new, colNames_train_new, colNames_test_new        


def Multi_Cross_Combination(colNames, features, OneHot=True):
    """
    多变量组合交叉衍生
    
    :param colNames: 参与交叉衍生的列名称
    :param features: 原始数据集
    :param OneHot: 是否进行独热编码
    
    :return: 交叉衍生后的新特征和新列名称
    """
    
    
    # 创建组合特征
    colNames_new = '&'.join([str(i) for i in colNames])
    features_new = features[colNames[0]].astype('str')

    for col in colNames[1:]:
        features_new = features_new + '&' + features[col].astype('str') 
    
    # 将组合特征转化为DataFrame
    features_new = pd.DataFrame(features_new, columns=[colNames_new])
    
    # 对新的特征列进行独热编码
    if OneHot == True:
        enc = preprocessing.OneHotEncoder()
        enc.fit_transform(features_new)
        colNames_new = cate_colName(enc, [colNames_new], drop=None)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=colNames_new)
        
    return features_new, colNames_new


def cate_colName(Transformer, category_cols, drop='if_binary'):
    """
    离散字段独热编码后字段名创建函数
    
    :param Transformer: 独热编码转化器
    :param category_cols: 输入转化器的离散变量
    :param drop: 独热编码转化器的drop参数
    """
    
    cate_cols_new = []
    col_value = Transformer.categories_
    
    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') & (len(col_value[i]) == 2):
            cate_cols_new.append(j)
        else:
            for f in col_value[i]:
                feature_name = j + '_' + f
                cate_cols_new.append(feature_name)
    return(cate_cols_new)
