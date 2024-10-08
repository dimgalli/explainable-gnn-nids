import argparse
import os
import re
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Preprocess raw data')
parser.add_argument('--benign-data', type=str, required=True, help='path to benign data')
parser.add_argument('--malicious-data', type=str, required=True, help='path to malicious data')
parser.add_argument('--train-data', type=str, required=True, help='path to save the train data')
parser.add_argument('--test-data', type=str, required=True, help='path to save the test data')

args = parser.parse_args()

if not os.path.exists(args.benign_data) or not os.path.isfile(args.benign_data):
    sys.exit('Path to benign data does not exist or is not a file')

if not os.path.exists(args.malicious_data) or not os.path.isfile(args.malicious_data):
    sys.exit('Path to malicious data does not exist or is not a file')

ben_data = pd.read_csv(args.benign_data)

mal_data = pd.read_csv(args.malicious_data)

data = pd.concat([ben_data, mal_data], ignore_index=True)

data = data.drop('ts', axis=1)

data = data[data.proto == 'tcp']
data.proto = data.proto.replace('tcp', 0)

src_ip_port = data.src_ip.astype(str) + ':' + data.src_port.astype(str)
data.insert(0, 'src_ip_port', src_ip_port)

dst_ip_port = data.dst_ip.astype(str) + ':' + data.dst_port.astype(str)
data.insert(1, 'dst_ip_port', dst_ip_port)

ip_src_type = data.src_ip.apply(lambda ip: 1 if (re.search('^10[.]', ip) or re.search('^172[.][1-3][678901][.]', ip) or re.search('^192[.]168[.]', ip)) else 0)
data.insert(2, 'ip_src_type', ip_src_type)
data = data.drop('src_ip', axis=1)

ip_dst_type = data.dst_ip.apply(lambda ip: 1 if (re.search('^10[.]', ip) or re.search('^172[.][1-3][678901][.]', ip) or re.search('^192[.]168[.]', ip)) else 0)
data.insert(3, 'ip_dst_type', ip_dst_type)
data = data.drop('dst_ip', axis=1)

src_port = data.src_port.astype(int)
src_port_wellknown = ((src_port >= 0) & (src_port <= 1023))
src_port_wellknown = src_port_wellknown.astype(int)
src_port_registered = ((src_port >= 1024) & (src_port <= 49151))
src_port_registered = src_port_registered.astype(int)
src_port_private = (src_port >= 49152)
src_port_private = src_port_private.astype(int)
data.insert(4, 'src_port_wellknown', src_port_wellknown)
data.insert(5, 'src_port_registered', src_port_registered)
data.insert(6, 'src_port_private', src_port_private)
data = data.drop('src_port', axis=1)

dst_port = data.dst_port.astype(int)
dst_port_wellknown = ((dst_port >= 0) & (dst_port <= 1023))
dst_port_wellknown = dst_port_wellknown.astype(int)
dst_port_registered = ((dst_port >= 1024) & (dst_port <= 49151))
dst_port_registered = dst_port_registered.astype(int)
dst_port_private = (dst_port >= 49152)
dst_port_private = dst_port_private.astype(int)
data.insert(7, 'dst_port_wellknown', dst_port_wellknown)
data.insert(8, 'dst_port_registered', dst_port_registered)
data.insert(9, 'dst_port_private', dst_port_private)
data = data.drop('dst_port', axis=1)

data = data.drop(data[~data.service.isin(['-', 'dns', 'http', 'ssl'])].index)
service = pd.get_dummies(data.service)
data.insert(10, '-', service['-'].astype(int))
data.insert(11, 'dns', service['dns'].astype(int))
data.insert(12, 'http', service['http'].astype(int))
data.insert(13, 'ssl', service['ssl'].astype(int))
data = data.drop('service', axis=1)

data = data.drop(data[~data.conn_state.isin(['S0', 'S1', 'SF', 'REJ', 'S2', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'RSTRH', 'SH', 'SHR', 'OTH'])].index)
conn_state = pd.get_dummies(data.conn_state)
data.insert(14, 'S0', conn_state['S0'].astype(int))
data.insert(15, 'S1', conn_state['S1'].astype(int))
data.insert(16, 'SF', conn_state['SF'].astype(int))
data.insert(17, 'REJ', conn_state['REJ'].astype(int))
data.insert(18, 'S2', conn_state['S2'].astype(int))
data.insert(19, 'S3', conn_state['S3'].astype(int))
data.insert(20, 'RSTO', conn_state['RSTO'].astype(int))
data.insert(21, 'RSTR', conn_state['RSTR'].astype(int))
data.insert(22, 'RSTOS0', conn_state['RSTOS0'].astype(int))
data.insert(23, 'RSTRH', conn_state['RSTRH'].astype(int))
data.insert(24, 'SH', conn_state['SH'].astype(int))
data.insert(25, 'SHR', conn_state['SHR'].astype(int))
data.insert(26, 'OTH', conn_state['OTH'].astype(int))
data = data.drop('conn_state', axis=1)

tot_bytes = data.src_bytes + data.dst_bytes
data.insert(36, 'tot_bytes', tot_bytes)

tot_pkts = data.src_pkts + data.dst_pkts
data.insert(37, 'tot_pkts', tot_pkts)

data = data.drop(['dns_query', 'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established', 'ssl_subject', 'ssl_issuer', 'http_trans_depth', 'http_method', 'http_uri', 'http_version', 'http_request_body_len', 'http_response_body_len', 'http_status_code', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice'], axis=1)

ben_data = data[data.label == 0].drop('type', axis=1)

mal_data = data[data.label == 1].drop('type', axis=1)

if len(ben_data) >= len(mal_data) * 10:
    ben_data = ben_data.sample(len(mal_data) * 10)
else:
    mal_data = mal_data.sample(len(ben_data) // 10)

data = pd.concat([ben_data, mal_data], ignore_index=True).sample(frac=1)

train, test = train_test_split(data, test_size=0.2)

train.to_csv(args.train_data, index=False)
test.to_csv(args.test_data, index=False)