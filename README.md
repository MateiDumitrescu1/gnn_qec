# QEC GNN Decoder

Minimal detector-graph GNN decoder baseline for a Stim surface-code memory experiment.

## Setup

```bash
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-End Run

```bash
python -m qec_gnn.generate_data --distance 3 --rounds 5 --p 0.005 --shots 30000 --out data/generated/raw_d3_r5_p005.npz
python -m qec_gnn.graph_builder --input data/generated/raw_d3_r5_p005.npz --max-nodes 64 --k 4 --out data/generated/graphs_d3_r5_p005.npz
python -m qec_gnn.train_gnn --data data/generated/graphs_d3_r5_p005.npz --tiny-overfit
python -m qec_gnn.train_gnn --data data/generated/graphs_d3_r5_p005.npz --epochs 30 --batch-size 128 --out outputs/models/gnn_d3_r5_p005.keras
python -m qec_gnn.eval_mwpm --raw-data data/generated/raw_d3_r5_p005.npz --out outputs/metrics/mwpm_d3_r5_p005.json
python -m qec_gnn.compare --gnn outputs/metrics/gnn_graphs_d3_r5_p005.json --mwpm outputs/metrics/mwpm_d3_r5_p005.json
```

The graph representation is one active detector event per node, node features are normalized `(x, y, t, detector_index)`, and edges are symmetric k-nearest neighbors in detector coordinate space with self-loops.
