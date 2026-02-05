# GNN-Based Construction Heuristic for CVRP

This implementation adds a Graph Neural Network (GNN) option for generating initial solutions in the AILS-CVRP solver.

## Overview

The GNN learns to construct high-quality initial CVRP solutions by training on real problem instances. The model is trained offline in Python/PyTorch and deployed in Java using Deep Java Library (DJL) for native performance.

### Architecture

- **Model**: Attention-based Encoder-Decoder (Kool et al. 2019)
- **Encoder**: 3-layer Graph Attention Network (128D embeddings, 8 attention heads)
- **Decoder**: Autoregressive decoder with capacity masking
- **Input Features**: Node coordinates (x, y), demand, distance to depot
- **Output**: Tour sequence respecting capacity constraints

### Performance

- **Inference Time**: 1-5ms per instance (Java native)
- **Training**: Offline in Python (no runtime overhead)
- **Deployment**: TorchScript model loaded by DJL

## Project Structure

```
AILS-CVRP/
├── python_ml/                          # Python training code
│   ├── gnn_model.py                   # GNN architecture
│   ├── train_gnn.py                   # Training script (REINFORCE)
│   ├── export_model.py                # Export to TorchScript
│   ├── generate_training_data.py      # Create dataset from .vrp files
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Python ML documentation
├── src/ML/
│   └── GNNConstructionHeuristic.java  # Java DJL inference wrapper
├── src/SearchMethod/
│   ├── ConstructSolution.java         # Integrated GNN + classical
│   └── Config.java                    # Configuration with GNN options
├── models/
│   └── gnn_construction.pt            # Trained model (after training)
├── scripts/
│   └── test_gnn_construction.sh       # Testing script
└── pom.xml                            # Maven with DJL dependencies
```

## Installation

### 1. Java Dependencies (Maven)

The `pom.xml` already includes DJL dependencies:
- `ai.djl:api` (v0.26.0)
- `ai.djl.pytorch:pytorch-engine`
- `ai.djl.pytorch:pytorch-native-auto`

Build the project:
```bash
mvn clean install
```

### 2. Python Environment (for training)

```bash
cd python_ml
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.1.0+
- torch-geometric 2.4.0+
- numpy, onnx

## Training the GNN

### Step 1: Generate Training Data

Convert existing .vrp files to training dataset:

```bash
cd python_ml
python generate_training_data.py --data_dir ../data --output_dir ./training_data
```

This processes all `.vrp` files in the `data/` directory and creates normalized feature matrices.

### Step 2: Train the Model

Train using REINFORCE policy gradient:

```bash
python train_gnn.py \
    --data_path training_data/cvrp_dataset.pt \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --embed_dim 128 \
    --num_heads 8 \
    --num_layers 3
```

Training options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--embed_dim`: Embedding dimension (default: 128)
- `--num_heads`: Attention heads (default: 8)
- `--num_layers`: Encoder layers (default: 3)

Checkpoints are saved in `checkpoints/`:
- `checkpoint_epoch_X.pt`: Regular checkpoints
- `best_model.pt`: Best model based on validation

### Step 3: Export to TorchScript

Convert the best model for Java deployment:

```bash
python export_model.py \
    --checkpoint checkpoints/best_model.pt \
    --output ../models/gnn_construction.pt
```

This creates a TorchScript file that DJL can load in Java.

## Usage

### Command Line

Enable GNN construction with command-line arguments:

```bash
mvn exec:java -Dexec.mainClass="Main.Main" -Dexec.args="\
    -i data/E-n101-k14.vrp \
    -t 60 \
    -s 42 \
    -useGNN true \
    -gnnModelPath models/gnn_construction.pt"
```

Parameters:
- `-useGNN true`: Enable GNN construction (default: false)
- `-gnnModelPath PATH`: Path to trained model (default: models/gnn_construction.pt)

### Programmatic Usage

```java
// Configure
Config config = new Config();
config.setUseGNN(true);
config.setGnnModelPath("models/gnn_construction.pt");

// Create solver
Instance instance = new Instance("data/E-n101-k14.vrp");
ConstructSolution constructor = new ConstructSolution(instance, config);

// Construct initial solution (automatically uses GNN if enabled)
Solution solution = new Solution(instance);
constructor.construct(solution);

// Cleanup
constructor.cleanup();
```

### Fallback Behavior

If GNN is enabled but fails (model not found, inference error), the solver automatically falls back to classical random construction.

## Testing

Run the comprehensive test script:

```bash
cd /Users/alejomonbar/Documents/GitHub/AILS-CVRP
./scripts/test_gnn_construction.sh
```

This script:
1. Tests baseline (classical construction) with seeds 42, 100, 200
2. Tests GNN construction with same seeds
3. Compares solution quality and inference time
4. Saves results to `results/gnn_test/`

## Implementation Details

### Java GNN Wrapper

`GNNConstructionHeuristic.java`:
- Loads TorchScript model via DJL
- Creates normalized feature matrices from CVRP instances
- Runs inference and converts output to routes
- Handles errors gracefully (returns null on failure)

### Integration with ConstructSolution

`ConstructSolution.java`:
- `construct()`: Main entry point - tries GNN first, falls back to classical
- `constructWithGNN()`: GNN-based construction
  * Get tour from GNN
  * Split into routes respecting capacity constraints
  * Calculate costs
- `constructClassical()`: Original random construction
- `cleanup()`: Release GNN resources

### Configuration

`Config.java`:
- `useGNN`: Enable/disable GNN (boolean)
- `gnnModelPath`: Path to TorchScript model (String)

## Performance Expectations

Based on similar implementations (Attention Model for CVRP):

### Initial Solution Quality
- **Small instances** (< 100 nodes): 1-5% better than random construction
- **Medium instances** (100-300 nodes): 5-10% improvement
- **Large instances** (> 300 nodes): 10-20% improvement

### Inference Time
- **Java native**: 1-5ms per instance
- **No iteration overhead**: One-shot construction
- **Scales linearly**: O(n²) with attention mechanism

### Training Time
- **100 epochs**: 1-4 hours (depending on dataset size and GPU)
- **Convergence**: Typically 50-100 epochs

## Comparison with RL Strategies

| Strategy | Type | Improvement | Overhead |
|----------|------|-------------|----------|
| Varphi RL | Q-Learning | +1,170 units | ~1ms per iteration |
| GNN Construction | Deep Learning | TBD (after training) | 1-5ms one-time |
| Combined | Hybrid | Potentially best | Minimal |

The GNN and RL strategies are **complementary**:
- **GNN**: Better initial solution (fewer iterations needed)
- **RL**: Better search strategy (higher quality final solution)
- **Combined**: Best of both worlds

## Troubleshooting

### Model Loading Errors

If you see "Failed to load GNN model":
1. Check model path: `models/gnn_construction.pt` exists?
2. Verify model format: Should be TorchScript (.pt), not checkpoint (.pth)
3. Check DJL dependencies: `mvn dependency:tree | grep djl`

### Inference Errors

If GNN construction fails:
1. Check instance size: Model trained on similar sizes?
2. Verify feature normalization: Are coordinates/demands reasonable?
3. Enable debug logging: Check `System.err` output

### Performance Issues

If inference is slow:
1. Check model size: Too many parameters?
2. Verify native library: DJL should use PyTorch native, not ONNX
3. Reduce model complexity: Fewer layers/smaller embeddings

## Future Improvements

1. **Beam Search**: Use beam search decoding for better solutions (slower)
2. **Multi-Task Learning**: Train on multiple problem variants
3. **Online Learning**: Fine-tune model during solving
4. **Hybrid Decoding**: Combine GNN with greedy/local search
5. **Ensemble**: Multiple models for robustness

## References

- Kool et al. (2019): "Attention, Learn to Solve Routing Problems!"
- Vinyals et al. (2015): "Pointer Networks"
- Deep Java Library: https://djl.ai/

## License

Same as parent project (see LICENSE file).
