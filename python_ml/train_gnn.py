"""
Training script for GNN CVRP Construction Heuristic

Uses REINFORCE with baseline for policy gradient training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
from gnn_model import CVRPGNNModel


class CVRPDataset(torch.utils.data.Dataset):
    """
    Dataset of CVRP instances
    
    Each instance: node_features [num_nodes, node_dim]
                  - [x, y, demand, distance_to_depot]
    """
    
    def __init__(self, data_dir, num_samples=10000, num_nodes=50, capacity=40):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.capacity = capacity
        self.data = []
        
        # Load pre-generated instances if available
        data_file = os.path.join(data_dir, f"cvrp_{num_nodes}.pt")
        if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            self.data = torch.load(data_file)
        else:
            print(f"Generating {num_samples} random instances...")
            self.generate_instances()
            os.makedirs(data_dir, exist_ok=True)
            torch.save(self.data, data_file)
            print(f"Saved data to {data_file}")
    
    def generate_instances(self):
        """Generate random CVRP instances"""
        for _ in range(self.num_samples):
            # Random coordinates [0, 1]
            coords = torch.rand(self.num_nodes, 2)
            coords[0] = torch.tensor([0.5, 0.5])  # Depot in center
            
            # Random demands [0.1, 1.0] scaled by capacity
            demands = torch.rand(self.num_nodes, 1) * 0.9 + 0.1
            demands[0] = 0  # Depot has no demand
            demands = demands * (self.capacity / demands.sum())  # Normalize
            
            # Distance to depot
            dist_to_depot = torch.norm(coords - coords[0:1], dim=1, keepdim=True)
            
            # Combine features
            features = torch.cat([coords, demands, dist_to_depot], dim=1)
            self.data.append(features)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def calculate_tour_length(node_features, tours):
    """
    Calculate total tour length
    
    Args:
        node_features: [batch_size, num_nodes, node_dim]
        tours: [batch_size, num_nodes] sequence of node indices
    
    Returns:
        lengths: [batch_size] total tour lengths
    """
    batch_size, num_nodes = tours.size()
    coords = node_features[:, :, :2]  # [B, N, 2]
    
    # Get coordinates of visited nodes in order
    batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(tours)
    tour_coords = coords[batch_idx, tours]  # [B, N, 2]
    
    # Add depot at start and end
    depot = coords[:, 0:1, :]  # [B, 1, 2]
    tour_with_depot = torch.cat([depot, tour_coords, depot], dim=1)  # [B, N+2, 2]
    
    # Calculate distances between consecutive nodes
    diff = tour_with_depot[:, 1:] - tour_with_depot[:, :-1]
    distances = torch.norm(diff, dim=2)  # [B, N+1]
    
    total_length = distances.sum(dim=1)  # [B]
    return total_length


def train_epoch(model, dataloader, optimizer, device, baseline_model=None):
    """Train for one epoch using REINFORCE"""
    model.train()
    total_loss = 0
    total_cost = 0
    
    for batch_idx, node_features in enumerate(tqdm(dataloader)):
        node_features = node_features.to(device)
        
        # Get node embeddings
        node_embeddings = model(node_features, decode=False)
        
        # Sample tours (with temperature for exploration)
        # For now, use greedy (TODO: add sampling)
        with torch.no_grad():
            tours = model(node_features, decode=True)
        
        # Calculate cost (tour length)
        cost = calculate_tour_length(node_features, tours)
        
        # Baseline
        if baseline_model is not None:
            with torch.no_grad():
                baseline_tours = baseline_model(node_features, decode=True)
                baseline_cost = calculate_tour_length(node_features, baseline_tours)
        else:
            baseline_cost = cost.mean()  # Use batch mean as baseline
        
        # REINFORCE loss: (cost - baseline) * log_prob
        # For now, simplified loss (TODO: proper policy gradient)
        loss = cost.mean()  # Supervised learning on greedy solutions
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_cost += cost.mean().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_cost = total_cost / len(dataloader)
    return avg_loss, avg_cost


def validate(model, dataloader, device):
    """Validate on test set"""
    model.eval()
    total_cost = 0
    
    with torch.no_grad():
        for node_features in dataloader:
            node_features = node_features.to(device)
            tours = model(node_features, decode=True)
            cost = calculate_tour_length(node_features, tours)
            total_cost += cost.mean().item()
    
    avg_cost = total_cost / len(dataloader)
    return avg_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='training_data', help='Data directory')
    parser.add_argument('--num_nodes', type=int, default=50, help='Number of nodes per instance')
    parser.add_argument('--num_samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading datasets...")
    train_dataset = CVRPDataset(args.data_dir, num_samples=args.num_samples, num_nodes=args.num_nodes)
    val_dataset = CVRPDataset(args.data_dir, num_samples=1000, num_nodes=args.num_nodes)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = CVRPGNNModel(
        node_dim=4,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_cost = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Train
        train_loss, train_cost = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Cost: {train_cost:.4f}")
        
        # Validate
        val_cost = validate(model, val_loader, device)
        print(f"Val Cost: {val_cost:.4f}")
        
        # Save best model
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cost': val_cost,
                'args': args
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Learning rate schedule
        scheduler.step()
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cost': val_cost,
                'args': args
            }, checkpoint_path)
    
    print(f"\nTraining complete! Best val cost: {best_val_cost:.4f}")


if __name__ == "__main__":
    main()
