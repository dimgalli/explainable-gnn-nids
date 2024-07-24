#!/bin/sh

for botnet in neris rbot virut menti murlo
do
    echo Botnet: $botnet

    echo Script: preprocessing.py
    python scripts/ctu/preprocessing.py --benign-data data/ctu/raw/benign.csv --malicious-data data/ctu/raw/malicious/$botnet.csv --training-data data/ctu/processed/train/$botnet.csv --testing-data data/ctu/processed/test/$botnet.csv
    
    echo Script: graphsage.py
    python scripts/ctu/graphsage.py --training-data data/ctu/processed/train/$botnet.csv --testing-data data/ctu/processed/test/$botnet.csv --model models/ctu/graphsage/$botnet.pth --scores scores/ctu/graphsage/$botnet.csv
    
    echo Script: gnnexplainer.py
    python scripts/ctu/gnnexplainer.py --testing-data data/ctu/processed/test/$botnet.csv --model models/ctu/graphsage/$botnet.pth --scores scores/ctu/gnnexplainer/$botnet.csv
    
    echo Script: captumexplainer.py
    python scripts/ctu/captumexplainer.py --testing-data data/ctu/processed/test/$botnet.csv --model models/ctu/graphsage/$botnet.pth --scores scores/ctu/captumexplainer/$botnet.csv
    
    echo Script: graphmaskexplainer.py
    python scripts/ctu/graphmaskexplainer.py --testing-data data/ctu/processed/test/$botnet.csv --model models/ctu/graphsage/$botnet.pth --scores scores/ctu/graphmaskexplainer/$botnet.csv
done

for attack in backdoor ddos dos injection password ransomware scanning xss
do
    echo Attack: $attack

    echo Script: preprocessing.py
    python scripts/ton/preprocessing.py --benign-data data/ton/raw/benign.csv --malicious-data data/ton/raw/malicious/$attack.csv --training-data data/ton/processed/train/$attack.csv --testing-data data/ton/processed/test/$attack.csv
    
    echo Script: graphsage.py
    python scripts/ton/graphsage.py --training-data data/ton/processed/train/$attack.csv --testing-data data/ton/processed/test/$attack.csv --model models/ton/graphsage/$attack.pth --scores scores/ton/graphsage/$attack.csv
    
    echo Script: gnnexplainer.py
    python scripts/ton/gnnexplainer.py --testing-data data/ton/processed/test/$attack.csv --model models/ton/graphsage/$attack.pth --scores scores/ton/gnnexplainer/$attack.csv
    
    echo Script: captumexplainer.py
    python scripts/ton/captumexplainer.py --testing-data data/ton/processed/test/$attack.csv --model models/ton/graphsage/$attack.pth --scores scores/ton/captumexplainer/$attack.csv
    
    echo Script: graphmaskexplainer.py
    python scripts/ton/graphmaskexplainer.py --testing-data data/ton/processed/test/$attack.csv --model models/ton/graphsage/$attack.pth --scores scores/ton/graphmaskexplainer/$attack.csv
done