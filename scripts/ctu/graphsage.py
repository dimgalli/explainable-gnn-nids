import argparse
import os
import sys

import dgl
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from dgl import LineGraph
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils.convert import from_dgl


def get_train_batch(data):
    ben_data = data[data.Label == 0].copy()
    mal_data = data[data.Label == 1].copy()

    while len(ben_data) >= 10000 and len(mal_data) >= 1000:
        ben_batch = ben_data.sample(10000)
        mal_batch = mal_data.sample(1000)

        ben_data = ben_data.drop(ben_batch.index)
        mal_data = mal_data.drop(mal_batch.index)

        batch = pd.concat([ben_batch, mal_batch], ignore_index=True).sample(frac=1)
        yield batch
    else:
        ben_batch = ben_data.copy()
        mal_batch = mal_data.copy()

        ben_data = ben_data.drop(ben_batch.index)
        mal_data = mal_data.drop(mal_batch.index)

        batch = pd.concat([ben_batch, mal_batch], ignore_index=True).sample(frac=1)
        yield batch


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
    G = nx.from_pandas_edgelist(data, source='SrcAddrPort', target='DstAddrPort', edge_attr=['x', 'Label'], create_using=nx.MultiGraph())
    G = G.to_directed()

    g = dgl.from_networkx(G, edge_attrs=['x', 'Label'])

    transform = LineGraph()
    lg = transform(g)

    return from_dgl(lg)


parser = argparse.ArgumentParser(description='Train and test GraphSAGE model')
parser.add_argument('--training-data', type=str, required=True, help='path to training data')
parser.add_argument('--testing-data', type=str, required=True, help='path to testing data')
parser.add_argument('--model', type=str, required=True, help='path to save the GraphSAGE model')
parser.add_argument('--scores', type=str, required=True, help='path to save the GraphSAGE model scores')

args = parser.parse_args()

if not os.path.exists(args.training_data) or not os.path.isfile(args.training_data):
    sys.exit('Path to training data does not exist or is not a file')

if not os.path.exists(args.testing_data) or not os.path.isfile(args.testing_data):
    sys.exit('Path to testing data does not exist or is not a file')

train_data = pd.read_csv(args.training_data)

feat = list(train_data)
feat.remove('SrcAddrPort')
feat.remove('DstAddrPort')
feat.remove('Label')

scaler = MinMaxScaler()
train_data[feat] = scaler.fit_transform(train_data[feat])

train_data.insert(42, 'x', train_data[feat].values.tolist())

model = GraphSAGE(
    in_channels=40,
    hidden_channels=128,
    num_layers=2,
    out_channels=2,
    dropout=0.2,
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 101):
    model.train()

    total_loss = total_nodes = 0
    for batch in get_train_batch(train_data):
        graph = to_graph(batch)
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(out, graph.Label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * graph.num_nodes
        total_nodes += graph.num_nodes
    
    print('Epoch {:03d}, Loss = {:.3f}'.format(epoch, total_loss / total_nodes))

torch.save(model, args.model)

test_data = pd.read_csv(args.testing_data)

feat = list(test_data)
feat.remove('SrcAddrPort')
feat.remove('DstAddrPort')
feat.remove('Label')

scaler = MinMaxScaler()
test_data[feat] = scaler.fit_transform(test_data[feat])

test_data.insert(42, 'x', test_data[feat].values.tolist())

amounts = [0, 1, 2, 5, 10, 20]
f1_scores = []
precision_scores = []
recall_scores = []

for amount in amounts:
    ben_data = test_data[test_data.Label == 0]
    mal_data = test_data[test_data.Label == 1].sample(len(test_data[test_data.Label == 1]) * 5 // 10)

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