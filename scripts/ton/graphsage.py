import argparse
import os
import sys

import dgl
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GraphSAGE
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
    G = nx.from_pandas_edgelist(data, source='src_ip_port', target='dst_ip_port', edge_attr=['x', 'label'], create_using=nx.MultiGraph())
    G = G.to_directed()

    g = dgl.from_networkx(G, edge_attrs=['x', 'label'])
    lg = g.line_graph(shared=True)

    data = from_dgl(lg)
    return data


parser = argparse.ArgumentParser(description='Train and test GraphSAGE model')
parser.add_argument('--train-data', type=str, required=True, help='path to train data')
parser.add_argument('--test-data', type=str, required=True, help='path to test data')
parser.add_argument('--model', type=str, required=True, help='path to save the GraphSAGE model')
parser.add_argument('--scores', type=str, required=True, help='path to save the GraphSAGE model scores')

args = parser.parse_args()

if not os.path.exists(args.train_data) or not os.path.isfile(args.train_data):
    sys.exit('Path to train data does not exist or is not a file')

if not os.path.exists(args.test_data) or not os.path.isfile(args.test_data):
    sys.exit('Path to test data does not exist or is not a file')

train_data = pd.read_csv(args.train_data)

feat = list(train_data)
feat.remove('src_ip_port')
feat.remove('dst_ip_port')
feat.remove('label')

scaler = MinMaxScaler()
train_data[feat] = scaler.fit_transform(train_data[feat])

train_data.insert(38, 'x', train_data[feat].values.tolist())

model = GraphSAGE(
    in_channels=36,
    hidden_channels=64,
    num_layers=2,
    out_channels=2,
    dropout=0.2
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 101):
    model.train()

    total_loss = total_examples = 0
    for batch in get_batch(train_data):
        data = to_graph(batch)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    
    print('Epoch {:03d}, Loss = {:.3f}'.format(epoch, total_loss / total_examples))

torch.save(model, args.model)

test_data = pd.read_csv(args.test_data)

feat = list(test_data)
feat.remove('src_ip_port')
feat.remove('dst_ip_port')
feat.remove('label')

scaler = MinMaxScaler()
test_data[feat] = scaler.fit_transform(test_data[feat])

test_data.insert(38, 'x', test_data[feat].values.tolist())

amounts = [0, 1, 2, 5, 10, 20]
f1_scores = []
precision_scores = []
recall_scores = []

for amount in amounts:
    ben_data = test_data[test_data.label == 0]
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