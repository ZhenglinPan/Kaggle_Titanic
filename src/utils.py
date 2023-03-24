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