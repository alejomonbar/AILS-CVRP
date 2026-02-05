# GNN Construction Heuristic for CVRP

This directory contains the Python implementation for training a Graph Neural Network (GNN) to generate initial solutions for the Capacitated Vehicle Routing Problem (CVRP).

## Architecture

The GNN uses an **Attention-based Encoder-Decoder** architecture inspired by recent papers:
- "Attention, Learn to Solve Routing Problems!" (Kool et al., 2019)
- "Learning Heuristics for the TSP by Policy Gradient" (Bello et al., 2016)

### Model Components:

1. **Graph Encoder**: 
   - Multi-head attention layers
   - Node embeddings: [x, y, demand, distance_to_depot]
   - Captures spatial and demand patterns

2. **Route Decoder**:
   - Autoregressive generation
   - Attention over unvisited nodes
   - Capacity-aware masking

3. **Training**:
   - REINFORCE with baseline
   - Supervised learning on optimal/near-optimal solutions

## Setup

```bash
cd python_ml
pip install -r requirements.txt
```

## Training

```bash
# Generate training data from CVRP instances
python generate_training_data.py --data_dir ../data --output_dir training_data

# Train GNN model
python train_gnn.py --data_dir training_data --epochs 100 --batch_size 32

# Export to ONNX/TorchScript for Java deployment
python export_model.py --model checkpoints/best_model.pt --output ../models/gnn_construction.pt
```

## Model Deployment

The trained model is loaded in Java using Deep Java Library (DJL):
- Fast inference (1-5ms per instance)
- Native Java performance
- No Python runtime required

## Files

- `gnn_model.py`: GNN architecture definition
- `train_gnn.py`: Training script
- `generate_training_data.py`: Data preparation
- `export_model.py`: Model export for Java
- `requirements.txt`: Python dependencies
