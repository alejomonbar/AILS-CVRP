"""
Generate training data from existing CVRP instances

Reads .vrp files from the data/ directory and creates a dataset
for GNN training. This uses real problem instances rather than
random synthetic data.
"""

import os
import re
import numpy as np
import torch
from pathlib import Path


def parse_vrp_file(filepath):
    """
    Parse a VRPTW format file and extract:
    - coordinates
    - demands
    - capacity
    - optimal/best known solution cost (if available from .sol file)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = {
        'name': '',
        'dimension': 0,
        'capacity': 0,
        'coordinates': [],
        'demands': [],
    }
    
    section = None
    for line in lines:
        line = line.strip()
        
        if line.startswith('NAME'):
            data['name'] = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            data['dimension'] = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            data['capacity'] = int(line.split(':')[1].strip())
        elif line == 'NODE_COORD_SECTION':
            section = 'coords'
        elif line == 'DEMAND_SECTION':
            section = 'demands'
        elif line == 'DEPOT_SECTION' or line == 'EOF':
            section = None
        elif section == 'coords' and line:
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                data['coordinates'].append([x, y])
        elif section == 'demands' and line:
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = float(parts[1])
                data['demands'].append(demand)
    
    return data


def parse_solution_file(filepath):
    """
    Parse .sol file to get best known solution cost
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'Cost' in line or 'cost' in line:
                # Extract number from line like "Cost: 12345" or "cost 12345"
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return float(numbers[0])
    return None


def normalize_instance(coords, demands, capacity):
    """
    Normalize instance data to [0, 1] range
    Returns: normalized coordinates, demands, and normalization factors
    """
    coords = np.array(coords)
    demands = np.array(demands)
    
    # Normalize coordinates
    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    range_coord = max_coord - min_coord
    range_coord[range_coord == 0] = 1  # Avoid division by zero
    
    normalized_coords = (coords - min_coord) / range_coord
    
    # Normalize demands
    normalized_demands = demands / capacity
    
    # Calculate distance to depot for each node
    depot = normalized_coords[0]
    dist_to_depot = np.linalg.norm(normalized_coords - depot, axis=1)
    max_dist = dist_to_depot.max()
    if max_dist > 0:
        dist_to_depot = dist_to_depot / max_dist
    
    return normalized_coords, normalized_demands, dist_to_depot


def create_node_features(coords, demands, capacity):
    """
    Create node feature matrix [num_nodes, 4]
    Features: [x, y, demand, dist_to_depot]
    """
    normalized_coords, normalized_demands, dist_to_depot = normalize_instance(
        coords, demands, capacity
    )
    
    # Stack features: [x, y, demand, dist_to_depot]
    features = np.column_stack([
        normalized_coords[:, 0],
        normalized_coords[:, 1],
        normalized_demands,
        dist_to_depot
    ])
    
    return torch.FloatTensor(features)


def process_vrp_files(data_dir='../data', output_dir='./training_data', max_instances=None):
    """
    Process all .vrp files in data directory and create training dataset
    
    Args:
        data_dir: Directory containing .vrp and .sol files
        output_dir: Directory to save processed data
        max_instances: Maximum number of instances to process (None = all)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    vrp_files = sorted(data_dir.glob('*.vrp'))
    
    if max_instances:
        vrp_files = vrp_files[:max_instances]
    
    print(f"Found {len(vrp_files)} .vrp files")
    
    instances = []
    
    for vrp_file in vrp_files:
        try:
            print(f"Processing {vrp_file.name}...", end=' ')
            
            # Parse VRP file
            data = parse_vrp_file(vrp_file)
            
            if len(data['coordinates']) < 2:
                print("SKIP (too small)")
                continue
            
            # Parse solution file if available
            sol_file = vrp_file.with_suffix('.sol')
            bks = parse_solution_file(sol_file)
            
            # Create features
            features = create_node_features(
                data['coordinates'],
                data['demands'],
                data['capacity']
            )
            
            instance = {
                'name': data['name'],
                'num_nodes': data['dimension'],
                'capacity': data['capacity'],
                'features': features,
                'bks': bks,  # Best known solution (optional)
                'coords': torch.FloatTensor(data['coordinates']),
                'demands': torch.FloatTensor(data['demands'])
            }
            
            instances.append(instance)
            print(f"OK (nodes={data['dimension']}, bks={bks})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Save dataset
    output_file = output_dir / 'cvrp_dataset.pt'
    torch.save(instances, output_file)
    print(f"\nSaved {len(instances)} instances to {output_file}")
    
    # Print statistics
    sizes = [inst['num_nodes'] for inst in instances]
    print(f"\nDataset statistics:")
    print(f"  Total instances: {len(instances)}")
    print(f"  Size range: {min(sizes)} - {max(sizes)} nodes")
    print(f"  Mean size: {np.mean(sizes):.1f} nodes")
    print(f"  Instances with BKS: {sum(1 for inst in instances if inst['bks'] is not None)}")
    
    return instances


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data from VRP instances')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing .vrp files')
    parser.add_argument('--output_dir', type=str, default='./training_data',
                        help='Directory to save processed data')
    parser.add_argument('--max_instances', type=int, default=None,
                        help='Maximum number of instances to process')
    
    args = parser.parse_args()
    
    instances = process_vrp_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_instances=args.max_instances
    )
