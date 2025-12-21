# #Kmeans基于距离 = √(ΔPC1² + ΔPC2² + ... + ΔPC9²)，而距离的几项中，PC1-PC4的方差占比最大，约为60%，
# # 每个维度在距离计算中权重相等，但业务信息集中在前几个PC:
# PC1/PC2 由原始重要特征（Age, SpendingScore 等）驱动
# PC8/PC9 可能只是次要特征组合因此在聚类结构主要由 PC1~PC3 决定。

# 先将聚类中心从PCA降9维的空间转换回原始空间，再映射到2D。
#需要先将聚类中心转回原空间，因为此时PCA是15→2D，而此时聚类中心是9D

#计算聚类内误差平方和（SSE）
#inertia_ = Σ(每个点到其所属簇中心的欧几里得距离²)

#PCA 的本质是用更少的维度，近似表达原始数据的结构,所以保留足够多的主成分，使得它们能解释原始数据 95% 的总方差
#如果更低的主成分数，会导致信息丢失，更多降维效果弱。
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read(file_path):
    """读取CSV文件并进行基础清洗，只提取前六列用于聚类分析"""
    df = pd.read_csv(file_path, encoding='utf-8')
    df2 = df.iloc[:,:6].copy()
    df2['Gender'] = df2['Gender'].map({'Male': 1, 'Female': 0}) 
    df2['Married'] = df2['Married'].map({'Yes': 1, 'No': 0})
    df2['SpendingScore'] = df2['SpendingScore'].map({'High': 2, 'Average': 1, 'Low': 0})
    df2['WorkExperience'].fillna(df2['WorkExperience'].mode()[0], inplace=True) 
    df2['Married'].fillna(df2['Married'].mode()[0], inplace=True) 
    df2['Profession'].fillna('unknown', inplace=True)
    print(f'样本数量：{len(df2)}')
    print(df2.head())
    return df2

def onehot(df, columns):
    """对指定列进行 One-Hot 编码"""
    print('开始进行 One-Hot 编码')
    df = pd.get_dummies(df, columns=columns, prefix=columns) 
    print('One-Hot 编码结束')
    return df

def normalize(df, columns, scaler=None): 
    """对指定列进行 Z-score 标准化"""
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler 

def visualize(X_pca, labels, centers_pca, k):
    """可视化聚类结果（基于 PCA 降维到 2D）"""
    plt.figure(figsize=(10, 8))
    colors = ['orange', 'green', 'blue', 'red', 'purple', 'brown', 'gray']
    for i in range(k):
        cluster_points = X_pca[labels == i] 
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='o', s=300, edgecolor='black', 
               linewidth=1, alpha=0.8, label='Centroids')
    plt.title("K-means聚类分析可视化")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def heatmap(original_df):
    """
    创建 Profession、WorkExperience 和 SpendingScore 关系的热力图
    :param original_df: 包含原始数据的DataFrame
    """
    df = original_df.copy()
    df['WorkExperience_Binned'] = pd.qcut(df['WorkExperience'], q=3, labels=['Low Experience', 'Medium Experience', 'High Experience'])
    crosstab = pd.crosstab(
        index=[df['Profession'], df['WorkExperience_Binned']],  
        columns=df['SpendingScore'], 
        normalize='index'
    ) * 100
    score_order = ['High', 'Average', 'Low']
    existing_scores = [score for score in score_order if score in crosstab.columns]
    crosstab = crosstab[existing_scores]
    plt.figure(figsize=(14, 10))
    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
    plt.title('职业、工作经验与消费评分关系热力图')
    plt.xlabel('消费评分')
    plt.ylabel('职业&工作经验')
    plt.tight_layout()
    plt.show()
    return crosstab

def printpca(X_pca, labels):
    df_pca = pd.DataFrame(
    X_pca, 
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])] 
     )
    df_pca['Cluster'] = labels
    pc_means = df_pca.groupby('Cluster').mean()
    pc_means.to_csv('各聚类pc表现新.csv', index=True)

def kmeans_pca(df, k, n_components_for_clustering=None, draw_plot=True):
    """
    执行标准化 -> PCA -> KMeans 聚类
    :param df: 输入 DataFrame
    :param k: 聚类数量
    :param n_components_for_clustering: 用于聚类的 PCA 维度（默认保留 95% 方差）
    :param draw_plot: 是否绘制 2D 可视化图
    """
    if k > len(df): 
        raise ValueError(f"n_samples={len(df)} should be >= n_clusters={k}")
    X = df.values
    print(X.shape) 
    if n_components_for_clustering is None:
        pca_clust = PCA(n_components=0.95)  
    else:
        pca_clust = PCA(n_components=n_components_for_clustering)
    X_pca_clust = pca_clust.fit_transform(X)
    print(f"PCA 降维后用于聚类的维度: {X_pca_clust.shape[1]}")
    feature_names = df.columns.tolist() 
    components_df = pd.DataFrame(
    pca_clust.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(pca_clust.n_components_)])
    components_df.round(4).to_csv('pca_components.csv')
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15) 
    labels = kmeans.fit_predict(X_pca_clust)
    inertia = kmeans.inertia_ 
    printpca(X_pca_clust, labels)
    if draw_plot:
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X)  
        centers_original = pca_clust.inverse_transform(kmeans.cluster_centers_) 
        centers_2d = pca_2d.transform(centers_original)  
        visualize(X_pca_2d, labels, centers_2d, k)
    return labels, inertia, kmeans, pca_clust ,X_pca_clust

if __name__ == "__main__":
    train = 'train-set.csv'
    df1 = pd.read_csv(train, encoding='utf-8') 
    df2 = read(train) 

    df2= onehot(df2, ['Profession'])
    df2, scaler = normalize(df2, ['Age', 'WorkExperience', 'SpendingScore'])
    print(df2.head())
    print(df2.iloc[0, :])
    # iner=[]
    # for k in range(1, 8):
    #     labels, inertia, model, pca_model = kmeans_pca(df2, k=k, draw_plot=False)
    #     iner.append(inertia)
    #     print(f"k={k}, Inertia (SSE): {inertia:.2f}")
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 8), iner, marker='o', linestyle='--', color='b')
    # plt.title('肘部法则：SSE 与 k 关系')
    # plt.xlabel('聚类数量 k')
    # plt.ylabel('SSE (Inertia)')
    # plt.xticks(range(1, 8))
    # plt.grid(True)
    # plt.show()  #通过肘部法则得到k=5
    k = min(5, len(df2))
    labels, inertia, model, pca_model,X_pca= kmeans_pca(df2, k=k, draw_plot=True)
    print(f"Inertia (SSE): {inertia:.2f}")
    df_result = df1.copy()
    df_result['Cluster'] = labels
    df_result.to_csv('聚类分类结果新.csv', index=False)
    print("结果已保存到聚类分类结果.csv")
    print("\n生成训练集（已有数据）Profession、WorkExperience 和 SpendingScore 的关系热力图")
    crosstab_multi =heatmap(df1)

    test='test-set.csv'
    df3=read(test)
    df3= onehot(df3, ['Profession'])
    df3, _ = normalize(df3, ['Age', 'WorkExperience', 'SpendingScore'], scaler) # 用训练集的scaler归一化测试集
    print(df3.head())
    # #传统预测，依据历史数据集静态预测
    # df3['Cluster'] = model.predict(pca_model.transform(df3))
    # df3.to_csv('测试集聚类分类结果.csv', index=False)
    # print("测试集结果已保存到测试集聚类分类结果.csv")
    # 基于XGboost的动态预测
    xgb_model = xgboost.XGBClassifier()
    xgb_model.fit(X_pca, labels) # 用训练集的特征和目标变量训练模型
    df3['Cluster'] = xgb_model.predict(pca_model.transform(df3))
    df3.to_csv('测试集聚类分类结果_xgb.csv', index=False)
    print("基于XGboost的测试集结果已保存到测试集聚类分类结果_xgb.csv")
    

    