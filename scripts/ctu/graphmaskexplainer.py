import argparse
import os
import sys

import dgl
import networkx as nx
import pandas as pd
import torch

from dgl import LineGraph
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GraphMaskExplainer
from torch_geometric.utils.convert import from_dgl


def get_test_batch(data):
    all_data = data.copy()
    while len(all_data) > 0:
        if len(all_data) > 11000:
            batch = all_data.sample(11000)
            all_data = all_data.drop(batch.index)
        else:
            batch = all_data.copy()
            all_data = all_data.drop(batch.index)
        yield batch


def to_graph(data):
    G = nx.from_pandas_edgelist(data, source='SrcAddrPort', target='DstAddrPort', edge_attr=['eid', 'x', 'Label'], create_using=nx.MultiGraph())
    G = G.to_directed()

    g = dgl.from_networkx(G, edge_attrs=['eid', 'x', 'Label'])

    transform = LineGraph()
    lg = transform(g)

    return from_dgl(lg)


parser = argparse.ArgumentParser(description='Train and test GraphMaskExplainer model')
parser.add_argument('--testing-data', type=str, required=True, help='path to testing data')
parser.add_argument('--model', type=str, required=True, help='path to GraphSAGE model')
parser.add_argument('--scores', type=str, required=True, help='path to save the GraphMaskExplainer model scores')

args = parser.parse_args()

if not os.path.exists(args.testing_data) or not os.path.isfile(args.testing_data):
    sys.exit('Path to testing data does not exist or is not a file')

if not os.path.exists(args.model) or not os.path.isfile(args.model):
    sys.exit('Path to GraphSAGE model does not exist or is not a file')

test_data = pd.read_csv(args.testing_data)

feat = list(test_data)
feat.remove('SrcAddrPort')
feat.remove('DstAddrPort')
feat.remove('Label')

scaler = MinMaxScaler()
test_data[feat] = scaler.fit_transform(test_data[feat])

test_data.insert(42, 'eid', test_data.index)

test_data.insert(43, 'x', test_data[feat].values.tolist())

model = torch.load(args.model)
model.eval()

model_config = ModelConfig(
    mode='binary_classification',
    task_level='node',
    return_type='raw',
)

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(num_layers=2, epochs=100),
    explanation_type='model',
    model_config=model_config,
    node_mask_type='object',
)

edge_identifier, features, labels, node_importance = [], [], [], []
for batch in get_test_batch(test_data):
    graph = to_graph(batch)
    explanation = explainer(graph.x, graph.edge_index)
    edge_identifier += graph.eid.tolist()
    features += graph.x.tolist()
    labels += graph.Label.tolist()
    node_importance += explanation.node_mask.squeeze().tolist()

edge_identifier = torch.tensor(edge_identifier)
features = torch.tensor(features)
labels = torch.tensor(labels)
node_importance = torch.tensor(node_importance)

mask = (labels == 1)
edge_identifier = edge_identifier[mask]
features = features[mask]
labels = labels[mask]
node_importance = node_importance[mask]

indices = torch.argsort(node_importance, descending=True)
edge_identifier = edge_identifier[indices]
features = features[indices]
labels = labels[indices]
node_importance = node_importance[indices]

topk = len(test_data[test_data.Label == 1]) * 5 // 10
edge_identifier = edge_identifier[:topk]
features = features[:topk]
labels = labels[:topk]
node_importance = node_importance[:topk]

amounts = [0, 1, 2, 5, 10, 20]
f1_scores = []
precision_scores = []
recall_scores = []

for amount in amounts:
    ben_data = test_data[test_data.Label == 0]
    mal_data = test_data.iloc[edge_identifier]

    adv_data = ben_data.sample(len(mal_data.SrcAddrPort.unique()) * amount, replace=True)
    adv_data.SrcAddrPort = list(mal_data.SrcAddrPort.unique()) * amount

    adv_data = pd.concat([adv_data, test_data], ignore_index=True)
    
    labels, predictions = [], []
    with torch.no_grad():
        for batch in get_test_batch(adv_data):
            graph = to_graph(batch)
            pred = model(graph.x, graph.edge_index).argmax(1)
            labels += graph.Label.tolist()
            predictions += pred.tolist()
    
    f1_scores.append(f1_score(labels, predictions))
    precision_scores.append(precision_score(labels, predictions))
    recall_scores.append(recall_score(labels, predictions))

scores = pd.DataFrame(data={'Amount': amounts, 'F1': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores})
scores.to_csv(args.scores, index=False)