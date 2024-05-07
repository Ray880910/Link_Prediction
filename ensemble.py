import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
import networkx as nx
import numpy as np
from math import log
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


# 讀取數據
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 創建圖
G = nx.Graph()
nodes = set(train_data['node1']).union(set(train_data['node2'])).union(set(test_data['node1'])).union(set(test_data['node2']))  # 獲取所有節點
G.add_nodes_from(nodes)

edges = [(row['node1'], row['node2']) for index, row in train_data.iterrows() if row['label'] == 1]
G.add_edges_from(edges)

def calculate_common_neighbor(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    return len(neighbors1.intersection(neighbors2))

def calculate_jaccard_coefficient(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    if not neighbors1 or not neighbors2:
        return 0
    intersection = len(neighbors1.intersection(neighbors2))
    union = len(neighbors1.union(neighbors2))
    return intersection / union

def calculate_shortest_path_length(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return -1  # 如果node不在圖中，返回-1
    try:
        shortest_path_length = nx.shortest_path_length(graph, source=node1, target=node2)
    except nx.NetworkXNoPath:
        shortest_path_length = -1  # 如果兩個node之間没有path，返回-1
    return shortest_path_length

def calculate_adamic_adar(row, graph):
    node1 = row['node1']
    node2 = row['node2']
    if node1 not in graph or node2 not in graph:
        return 0
    common_neighbors = set(graph.neighbors(node1)).intersection(set(graph.neighbors(node2)))
    adamic_adar = sum(1 / (log(len(set(graph.neighbors(neighbor))))) for neighbor in common_neighbors)
    return adamic_adar

def generate_features(data):
    #計算 common neighbor
    data['common_neighbor'] = data.apply(lambda row: calculate_common_neighbor(row, G), axis=1)
    
    #計算 Jaccard's coefficient
    data['jaccard_coefficient'] = data.apply(lambda row: calculate_jaccard_coefficient(row, G), axis=1)
    
    #計算 最短路徑
    data['shortest_path_length'] = data.apply(lambda row: calculate_shortest_path_length(row, G), axis=1)
    
    #計算 adamic_adar
    data['adamic_adar'] = data.apply(lambda row: calculate_adamic_adar(row, G), axis=1)

    return data

train_data = generate_features(train_data)

X = train_data[['common_neighbor', 'jaccard_coefficient', 'shortest_path_length', 'adamic_adar', 'node1', 'node2']]
y = train_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建 Random Forest 模型
rf_model = RandomForestClassifier()

# 創建 SVM 模型
svm_model = SVC(probability=True)

# 創建 Logistic Regression 模型
lr_model = LogisticRegression()

# 創建 Voting Classifier 集成模型
ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('lr', lr_model)], voting='soft')

# 擬合模型
ensemble_model.fit(X_train, y_train)

# 進行預測
predictions = ensemble_model.predict(X_test)

# 測試集特徵生成
test_data = generate_features(test_data)

# 進行預測
X2 = test_data[['common_neighbor', 'jaccard_coefficient', 'shortest_path_length', 'adamic_adar', 'node1', 'node2']]
predictions2 = ensemble_model.predict(X2)

# 輸出結果到csv
sample_submission = pd.DataFrame({'idx': test_data['idx'], 'ans': predictions2})
sample_submission.to_csv('ensemble.csv', index=False)