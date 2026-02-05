"""
Export trained GNN model to TorchScript for Java deployment

TorchScript is PyTorch's serialization format that can be loaded in Java via DJL
"""

import torch
import argparse
import os
from gnn_model import CVRPGNNModel


def export_to_torchscript(model, output_path, num_nodes=50):
    """
    Export model to TorchScript format
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save .pt file
        num_nodes: Number of nodes for tracing (example input size)
    """
    model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(1, num_nodes, 4)  # [batch=1, nodes, features]
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    print(f"Saving to {output_path}")
    traced_model.save(output_path)
    
    # Verify
    print("Verifying model...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        original_output = model(example_input)
        loaded_output = loaded_model(example_input)
        
        if isinstance(original_output, torch.Tensor):
            diff = (original_output - loaded_output).abs().max().item()
            print(f"Max difference: {diff}")
            assert diff < 1e-5, "Model outputs don't match!"
    
    print("Export successful!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output path for exported model')
    parser.add_argument('--num_nodes', type=int, default=50, help='Number of nodes for tracing')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    
    # Get model args from checkpoint
    model_args = checkpoint.get('args', None)
    if model_args:
        embedding_dim = model_args.embedding_dim
        num_heads = model_args.num_heads
        num_layers = model_args.num_layers
    else:
        # Defaults
        embedding_dim = 128
        num_heads = 8
        num_layers = 3
    
    # Create model
    model = CVRPGNNModel(
        node_dim=4,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation cost: {checkpoint.get('val_cost', 'N/A')}")
    
    # Export
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_to_torchscript(model, args.output, args.num_nodes)
    
    # Print model info
    file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    print(f"\nModel file size: {file_size:.2f} MB")
    print(f"Ready for Java deployment with DJL!")
    
    print("\n=== Java Usage ===")
    print("Model model = Model.newInstance(\"gnn_cvrp\");")
    print(f"model.load(Paths.get(\"{args.output}\"));")
    print("Predictor<NDList, NDList> predictor = model.newPredictor(translator);")


if __name__ == "__main__":
    main()
