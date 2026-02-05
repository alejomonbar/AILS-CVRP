"""
M1 MacBook Optimized Training Script for GNN CVRP Construction

Features:
- MPS (Metal Performance Shaders) acceleration for M1 GPU
- Smaller model size (~100K parameters vs 500K)
- Memory-efficient batch processing
- Faster training with early stopping
- Progress tracking and ETA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from gnn_model import CVRPGNNModel


def get_device():
    """Get best available device for M1 Mac"""
    if torch.backends.mps.is_available():
        print("\n🚀 Using M1 GPU (MPS) acceleration")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print("\n🚀 Using CUDA GPU acceleration")
        return torch.device('cuda')
    else:
        print("\n⚠️  Using CPU (slower, but works)")
        return torch.device('cpu')


class CVRPDataset(Dataset):
    """
    Lightweight CVRP dataset for M1 training
    Generates instances on-the-fly to save memory
    """
    
    def __init__(self, num_samples=100, num_nodes=30, capacity=40):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.capacity = capacity
        # Pre-generate random seeds for reproducibility
        self.seeds = torch.randint(0, 1000000, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Use seed for reproducibility
        torch.manual_seed(self.seeds[idx].item())
        
        # Random coordinates [0, 1]
        coords = torch.rand(self.num_nodes, 2)
        coords[0] = torch.tensor([0.5, 0.5])  # Depot in center
        
        # Random demands [1, 9]
        demands = torch.randint(1, 10, (self.num_nodes,)).float()
        demands[0] = 0  # Depot has no demand
        
        # Normalize demands
        demands = demands / self.capacity
        
        # Calculate distance to depot
        depot = coords[0]
        dist_to_depot = torch.norm(coords - depot, dim=1)
        max_dist = dist_to_depot.max()
        if max_dist > 0:
            dist_to_depot = dist_to_depot / max_dist
        
        # Create features [num_nodes, 4]: x, y, demand, dist_to_depot
        features = torch.cat([
            coords,
            demands.unsqueeze(1),
            dist_to_depot.unsqueeze(1)
        ], dim=1)
        
        return features, coords, demands * self.capacity


def calculate_tour_length(coords, tour):
    """Calculate total tour length"""
    tour_coords = coords[tour]
    # Add return to depot
    tour_coords_shifted = torch.cat([tour_coords[1:], tour_coords[0:1]], dim=0)
    distances = torch.norm(tour_coords - tour_coords_shifted, dim=1)
    return distances.sum()


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch using REINFORCE"""
    model.train()
    total_cost = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_features, batch_coords, batch_demands in pbar:
        batch_features = batch_features.to(device)
        batch_coords = batch_coords.to(device)
        
        batch_size = batch_features.size(0)
        
        # Forward pass: get log probabilities and tours
        log_probs_list = []
        tours_list = []
        
        for i in range(batch_size):
            features = batch_features[i]
            coords = batch_coords[i]
            
            # Get tour from model (sampling during training)
            tour, log_probs = model(features.unsqueeze(0), return_log_probs=True, greedy=False)
            tour = tour.squeeze(0)
            
            tours_list.append(tour)
            log_probs_list.append(log_probs)
            
            # Calculate cost (negative reward)
            cost = calculate_tour_length(coords, tour)
            total_cost += cost.item()
        
        # REINFORCE loss
        loss = 0
        for i, (log_probs, tour) in enumerate(zip(log_probs_list, tours_list)):
            coords = batch_coords[i]
            cost = calculate_tour_length(coords, tour)
            
            # Use cost as negative reward
            loss += (log_probs * cost).mean()
        
        loss = loss / batch_size
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        pbar.set_postfix({'loss': f'{loss.item():.2f}'})
    
    return total_cost / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model (greedy decoding)"""
    model.eval()
    total_cost = 0
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch_features, batch_coords, batch_demands in pbar:
        batch_features = batch_features.to(device)
        batch_coords = batch_coords.to(device)
        
        batch_size = batch_features.size(0)
        
        for i in range(batch_size):
            features = batch_features[i]
            coords = batch_coords[i]
            
            # Greedy decoding for validation
            tour = model(features.unsqueeze(0), greedy=True).squeeze(0)
            
            # Calculate cost
            cost = calculate_tour_length(coords, tour)
            total_cost += cost.item()
        
        pbar.set_postfix({'avg_cost': f'{total_cost / ((pbar.n + 1) * batch_size):.2f}'})
    
    return total_cost / len(dataloader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN for CVRP (M1 Optimized)')
    
    # M1-optimized defaults
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs (M1 optimized: 30)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (M1 optimized: 4)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (M1 optimized: 0.0005)')
    
    # Model architecture (smaller for M1)
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension (M1: 64 vs 128)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Attention heads (M1: 4 vs 8)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Encoder layers (M1: 2 vs 3)')
    
    # Dataset
    parser.add_argument('--train_samples', type=int, default=200,
                        help='Training samples (M1: 200)')
    parser.add_argument('--val_samples', type=int, default=50,
                        help='Validation samples (M1: 50)')
    parser.add_argument('--num_nodes', type=int, default=30,
                        help='Nodes per instance (M1: 30)')
    
    # Training settings
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🧠 GNN CVRP Training (M1 MacBook Optimized)")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    print(f"   Training samples: {args.train_samples}")
    print(f"   Validation samples: {args.val_samples}")
    print(f"   Instance size: {args.num_nodes} nodes")
    
    train_dataset = CVRPDataset(
        num_samples=args.train_samples,
        num_nodes=args.num_nodes,
        capacity=40
    )
    
    val_dataset = CVRPDataset(
        num_samples=args.val_samples,
        num_nodes=args.num_nodes,
        capacity=40
    )
    
    # Create dataloaders (num_workers=0 for M1 MPS compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Disable for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print(f"\n🧠 Creating GNN model...")
    print(f"   Embed dim: {args.embed_dim}")
    print(f"   Attention heads: {args.num_heads}")
    print(f"   Encoder layers: {args.num_layers}")
    
    model = CVRPGNNModel(
        input_dim=4,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    memory_mb = num_params * 4 / 1024 / 1024
    print(f"   Parameters: {num_params:,} (~{num_params/1000:.0f}K)")
    print(f"   Model size: ~{memory_mb:.1f} MB")
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Training loop
    print(f"\n🏋️  Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Early stopping patience: {args.patience}")
    print()
    
    best_val_cost = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_cost = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_cost = validate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_cost)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate times
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (args.epochs - epoch - 1)
        
        # Print summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train cost: {train_cost:.2f}")
        print(f"   Val cost:   {val_cost:.2f} {'✨ NEW BEST!' if val_cost < best_val_cost else ''}")
        print(f"   LR:         {current_lr:.6f}")
        print(f"   Time:       {epoch_time:.1f}s")
        print(f"   Elapsed:    {elapsed/60:.1f}min")
        print(f"   ETA:        {eta/60:.1f}min")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_cost': train_cost,
                'val_cost': val_cost,
                'args': vars(args)
            }, checkpoint_path)
            print(f"   💾 Checkpoint: {checkpoint_path.name}")
        
        # Save best model
        if val_cost < best_val_cost:
            best_val_cost = val_cost
            patience_counter = 0
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_cost': train_cost,
                'val_cost': val_cost,
                'args': vars(args)
            }, best_path)
            print(f"   ⭐ Best model saved! (Val: {val_cost:.2f})")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping (no improvement for {args.patience} epochs)")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"   Best val cost: {best_val_cost:.2f}")
    print(f"   Total time:    {total_time/60:.1f} minutes")
    print(f"   Best model:    {checkpoint_dir / 'best_model.pt'}")
    
    print(f"\n📌 Next Step: Export model for Java")
    print(f"   python export_model.py \\")
    print(f"       --checkpoint {checkpoint_dir / 'best_model.pt'} \\")
    print(f"       --output ../models/gnn_construction.pt")
    print()


if __name__ == '__main__':
    main()
