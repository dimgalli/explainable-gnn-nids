import argparse
import os
import sys

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GraphMaskExplainer
from torch_geometric.utils.convert import from_dgl


def get_batch(data):
    all_data = data.copy()
    while len(all_data) > 0:
        if len(all_data) >= 5000:
            batch = all_data.sample(5000)
            all_data = all_data.drop(batch.index)
            yield batch
        else:
            batch = all_data.copy()
            all_data = all_data.drop(batch.index)
            yield batch


def to_graph(data):
    G = nx.from_pandas_edgelist(data, source='src_ip_port', target='dst_ip_port', edge_attr=['i', 'x', 'label'], create_using=nx.MultiGraph())
    G = G.to_directed()

    g = dgl.from_networkx(G, edge_attrs=['i', 'x', 'label'])
    lg = g.line_graph(shared=True)

    data = from_dgl(lg)
    return data


parser = argparse.ArgumentParser(description='Train and test GraphMaskExplainer algorithm')
parser.add_argument('--test-data', type=str, required=True, help='path to test data')
parser.add_argument('--model', type=str, required=True, help='path to GraphSAGE model')
parser.add_argument('--scores', type=str, required=True, help='path to save the GraphMaskExplainer algorithm scores')

args = parser.parse_args()

if not os.path.exists(args.test_data) or not os.path.isfile(args.test_data):
    sys.exit('Path to test data does not exist or is not a file')

if not os.path.exists(args.model) or not os.path.isfile(args.model):
    sys.exit('Path to GraphSAGE model does not exist or is not a file')

test_data = pd.read_csv(args.test_data)

feat = list(test_data)
feat.remove('src_ip_port')
feat.remove('dst_ip_port')
feat.remove('label')

scaler = MinMaxScaler()
test_data[feat] = scaler.fit_transform(test_data[feat])

test_data.insert(38, 'i', test_data.index)
test_data.insert(39, 'x', test_data[feat].values.tolist())

model = torch.load(args.model)

model_config = ModelConfig(
    mode='binary_classification',
    task_level='node',
    return_type='raw'
)

explainer = Explainer(
    model=model,
    algorithm=GraphMaskExplainer(2, epochs=100),
    explanation_type='model',
    model_config=model_config,
    node_mask_type='object',
    edge_mask_type=None
)

edge_identifiers, labels, node_importances, predictions = [], [], [], []
for batch in get_batch(test_data):
    data = to_graph(batch)
    explanation = explainer(data.x, data.edge_index)
    prediction = explainer.get_prediction(data.x, data.edge_index).argmax(1)

    edge_identifiers += data.i.tolist()
    labels += data.label.tolist()
    node_importances += explanation.node_mask.squeeze().tolist()
    predictions += prediction.tolist()

edge_identifiers = np.array(edge_identifiers)
labels = np.array(labels)
node_importances = np.array(node_importances)
predictions = np.array(predictions)

mask = (labels == 0) & (predictions == 0)
edge_identifiers = edge_identifiers[mask]
labels = labels[mask]
node_importances = node_importances[mask]
predictions = predictions[mask]

index_array = np.argsort(node_importances)[::-1]
edge_identifiers = edge_identifiers[index_array]
labels = labels[index_array]
node_importances = node_importances[index_array]
predictions = predictions[index_array]

topk = len(edge_identifiers) // 2
edge_identifiers = edge_identifiers[:topk]
labels = labels[:topk]
node_importances = node_importances[:topk]
predictions = predictions[:topk]

amounts = [0, 1, 2, 5, 10, 20]
f1_scores = []
precision_scores = []
recall_scores = []

for amount in amounts:
    ben_data = test_data.loc[edge_identifiers]
    mal_data = test_data[test_data.label == 1]

    src_ip_port = mal_data.src_ip_port.unique()

    adv_data = ben_data.sample(amount * len(src_ip_port), replace=True)
    adv_data.src_ip_port = amount * list(src_ip_port)

    adv_data = pd.concat([adv_data, test_data], ignore_index=True)

    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for batch in get_batch(adv_data):
            data = to_graph(batch)
            prediction = model(data.x, data.edge_index).argmax(1)
            
            labels += data.label.tolist()
            predictions += prediction.tolist()
    
    f1_scores.append(f1_score(labels, predictions))
    precision_scores.append(precision_score(labels, predictions))
    recall_scores.append(recall_score(labels, predictions))

scores = pd.DataFrame(data={'Amount': amounts, 'F1': f1_scores, 'Precision': precision_scores, 'Recall': recall_scores})
scores.to_csv(args.scores, index=False)