"""
Graph Neural Network for CVRP Construction Heuristic

Architecture based on:
- "Attention, Learn to Solve Routing Problems!" (Kool et al., 2019)
- Encoder-Decoder with Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionEncoder(nn.Module):
    """
    Graph Attention Encoder for CVRP instances
    
    Input: Node features [batch_size, num_nodes, node_dim]
           - x, y coordinates (normalized)
           - demand (normalized by capacity)
           - distance to depot (normalized)
    
    Output: Node embeddings [batch_size, num_nodes, embedding_dim]
    """
    
    def __init__(self, node_dim=4, embedding_dim=128, num_heads=8, num_layers=3):
        super(GraphAttentionEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initial embedding
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.ReLU(),
                nn.Linear(4 * embedding_dim, embedding_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm1 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        
    def forward(self, node_features):
        """
        Args:
            node_features: [batch_size, num_nodes, node_dim]
        Returns:
            node_embeddings: [batch_size, num_nodes, embedding_dim]
        """
        # Initial embedding
        h = self.init_embed(node_features)  # [B, N, embedding_dim]
        
        # Apply attention layers
        for i in range(self.num_layers):
            # Multi-head attention with residual
            h_att = self.attention_layers[i](h, h, h)
            h = self.norm1[i](h + h_att)
            
            # Feed-forward with residual
            h_ff = self.feed_forward[i](h)
            h = self.norm2[i](h + h_ff)
        
        return h


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [B, N, embedding_dim]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head: [B, N, embedding_dim] -> [B, num_heads, N, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Reshape back: [B, num_heads, N, head_dim] -> [B, N, embedding_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        
        # Output projection
        output = self.W_o(context)
        
        return output


class AttentionDecoder(nn.Module):
    """
    Attention-based Decoder for route construction
    
    Autoregressively selects next customer to visit
    """
    
    def __init__(self, embedding_dim=128, num_heads=8):
        super(AttentionDecoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Context embedding (current vehicle state)
        self.context_embed = nn.Linear(embedding_dim + 2, embedding_dim)  # +2 for [remaining_capacity, current_load]
        
        # Attention for node selection
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        
        # Glimpse attention (multi-head)
        self.glimpse = MultiHeadAttention(embedding_dim, num_heads)
        
        # Final projection
        self.pointer = nn.Linear(embedding_dim, 1)
        
    def forward(self, node_embeddings, context, mask):
        """
        Args:
            node_embeddings: [batch_size, num_nodes, embedding_dim] from encoder
            context: [batch_size, embedding_dim + 2] current vehicle state
            mask: [batch_size, num_nodes] feasibility mask (0 = infeasible, 1 = feasible)
        
        Returns:
            logits: [batch_size, num_nodes] unnormalized scores
            probs: [batch_size, num_nodes] selection probabilities
        """
        batch_size, num_nodes, _ = node_embeddings.size()
        
        # Embed context
        context_emb = self.context_embed(context).unsqueeze(1)  # [B, 1, embedding_dim]
        
        # Glimpse attention
        glimpse_q = context_emb.repeat(1, num_nodes, 1)
        glimpse_out = self.glimpse(glimpse_q, node_embeddings, node_embeddings)
        
        # Compatibility scores
        query = self.W_q(context_emb)  # [B, 1, embedding_dim]
        keys = self.W_k(node_embeddings)  # [B, N, embedding_dim]
        
        logits = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.embedding_dim)  # [B, N]
        
        # Apply mask (infeasible nodes get -inf logits)
        logits = logits.masked_fill(mask == 0, -1e9)
        
        # Probabilities
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs


class CVRPGNNModel(nn.Module):
    """
    Complete GNN model for CVRP Construction
    
    Given a CVRP instance, generates an initial solution (routes)
    """
    
    def __init__(self, node_dim=4, embedding_dim=128, num_heads=8, num_layers=3):
        super(CVRPGNNModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Encoder: graph -> node embeddings
        self.encoder = GraphAttentionEncoder(node_dim, embedding_dim, num_heads, num_layers)
        
        # Decoder: node embeddings -> route construction
        self.decoder = AttentionDecoder(embedding_dim, num_heads)
        
    def forward(self, node_features, decode=True):
        """
        Args:
            node_features: [batch_size, num_nodes, node_dim]
                          node 0 is depot, nodes 1..N are customers
            decode: if True, perform greedy decoding
        
        Returns:
            if decode=True: tours [batch_size, num_nodes] sequence of node indices
            if decode=False: node_embeddings for training
        """
        # Encode graph
        node_embeddings = self.encoder(node_features)
        
        if not decode:
            return node_embeddings
        
        # Greedy decoding
        batch_size, num_nodes, _ = node_features.size()
        tours = []
        
        # Start from depot (scalar for all batches)
        current_nodes = torch.zeros(batch_size, dtype=torch.long, device=node_features.device)
        visited = torch.zeros(batch_size, num_nodes, device=node_features.device)
        visited[:, 0] = 1  # Depot always visited
        
        remaining_capacity = torch.ones(batch_size, 1, device=node_features.device)
        current_load = torch.zeros(batch_size, 1, device=node_features.device)
        
        for step in range(num_nodes - 1):
            # Context: last node embedding + capacity info (remaining_capacity + current_load)
            current_emb = node_embeddings[torch.arange(batch_size), current_nodes, :]  # [B, embedding_dim]
            context = torch.cat([current_emb, remaining_capacity, current_load], dim=-1)  # [B, embedding_dim + 2]
            
            # Get selection probabilities
            mask = (1 - visited) * (node_features[:, :, 2] <= remaining_capacity)  # Feasibility mask
            mask[:, 0] = 1  # Depot always feasible
            
            _, probs = self.decoder(node_embeddings, context, mask)
            
            # Select next node (greedy)
            next_nodes = torch.argmax(probs, dim=-1)  # [B]
            tours.append(next_nodes)
            
            # Update state
            visited[torch.arange(batch_size), next_nodes] = 1
            demand = node_features[torch.arange(batch_size), next_nodes, 2].unsqueeze(1)  # [B, 1]
            
            # If returning to depot, reset capacity and load
            is_depot = (next_nodes == 0).float().unsqueeze(1)  # [B, 1]
            remaining_capacity = is_depot + (1 - is_depot) * (remaining_capacity - demand)
            current_load = (1 - is_depot) * (current_load + demand)
            
            current_nodes = next_nodes
        
        tours = torch.stack(tours, dim=1)
        return tours
    
    def beam_search(self, node_features, beam_width=5):
        """
        Beam search for better solutions (slower but higher quality)
        """
        # TODO: Implement beam search decoding
        pass


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    num_nodes = 20
    node_dim = 4
    
    # Random instance
    node_features = torch.randn(batch_size, num_nodes, node_dim)
    node_features[:, 0, :] = 0  # Depot at origin with zero demand
    
    model = CVRPGNNModel(node_dim=node_dim, embedding_dim=128, num_heads=8, num_layers=3)
    
    # Forward pass
    tours = model(node_features, decode=True)
    print(f"Input shape: {node_features.shape}")
    print(f"Output tours shape: {tours.shape}")
    print(f"Tours: {tours}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
