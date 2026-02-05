#!/bin/bash

# Quick Start Training for M1 MacBook
# Optimized settings for M1 hardware

echo "======================================"
echo "🚀 GNN Training - M1 MacBook Optimized"
echo "======================================"
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Navigate to python_ml directory
cd "$(dirname "$0")/../python_ml" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio -q
pip install numpy tqdm -q
echo "✅ Dependencies installed"

echo ""
echo "======================================"
echo "🧠 Starting GNN Training"
echo "======================================"
echo ""
echo "⚙️  M1 Optimized Settings:"
echo "   • Model: 64D embeddings, 4 heads, 2 layers (~100K params)"
echo "   • Data: 200 train + 50 val instances (30 nodes each)"
echo "   • Training: 30 epochs, batch size 4"
echo "   • Expected time: 15-30 minutes on M1"
echo "   • Memory usage: ~2-4 GB"
echo ""

read -p "Press Enter to start training (or Ctrl+C to cancel)..."

# Run training
python3 train_gnn_m1.py \
    --epochs 30 \
    --batch_size 4 \
    --lr 0.0005 \
    --embed_dim 64 \
    --num_heads 4 \
    --num_layers 2 \
    --train_samples 200 \
    --val_samples 50 \
    --num_nodes 30 \
    --patience 10

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ Training Complete!"
    echo "======================================"
    echo ""
    echo "📌 Next Steps:"
    echo ""
    echo "1. Export model for Java:"
    echo "   python3 export_model.py \\"
    echo "       --checkpoint checkpoints/best_model.pt \\"
    echo "       --output ../models/gnn_construction.pt"
    echo ""
    echo "2. Test the model:"
    echo "   cd .."
    echo "   ./scripts/test_gnn_construction.sh"
    echo ""
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
