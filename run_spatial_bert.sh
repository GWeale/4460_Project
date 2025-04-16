#!/bin/bash
# Run Spatial-BERT for CODEX Melanoma Data
# This script runs the full pipeline for the Spatial-BERT model

# Default parameters
METADATA_PATH="metadata.csv"
CELL_DATA_PATH="Melanoma_data.csv"
OUTPUT_DIR="output"
EVAL_DIR="evaluation"
OS_THRESHOLD="" # Empty means use median
K_NEIGHBORS=20
WINDOWS_PER_SAMPLE=500
HIDDEN_DIM=256
NUM_HEADS=8
NUM_LAYERS=6
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0001
EARLY_STOPPING=10
USE_BATCH_CORRECTION=false
USE_GLOBAL_FEATURES=false
DEVICE="cuda"  # Use "cpu" if no GPU available

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metadata)
            METADATA_PATH="$2"
            shift 2
            ;;
        --cell_data)
            CELL_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --eval_dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --os_threshold)
            OS_THRESHOLD="$2"
            shift 2
            ;;
        --k_neighbors)
            K_NEIGHBORS="$2"
            shift 2
            ;;
        --windows)
            WINDOWS_PER_SAMPLE="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --num_heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --num_layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --early_stopping)
            EARLY_STOPPING="$2"
            shift 2
            ;;
        --use_batch_correction)
            USE_BATCH_CORRECTION=true
            shift
            ;;
        --use_global_features)
            USE_GLOBAL_FEATURES=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --metadata PATH           Path to metadata CSV (default: metadata.csv)"
            echo "  --cell_data PATH          Path to cell data CSV (default: Melanoma_data.csv)"
            echo "  --output_dir DIR          Output directory for training (default: output)"
            echo "  --eval_dir DIR            Output directory for evaluation (default: evaluation)"
            echo "  --os_threshold VALUE      Threshold for high/low survival (default: median)"
            echo "  --k_neighbors VALUE       Number of neighbors per window (default: 20)"
            echo "  --windows VALUE           Number of windows per sample (default: 500)"
            echo "  --hidden_dim VALUE        Hidden dimension for model (default: 256)"
            echo "  --num_heads VALUE         Number of attention heads (default: 8)"
            echo "  --num_layers VALUE        Number of transformer layers (default: 6)"
            echo "  --batch_size VALUE        Batch size (default: 32)"
            echo "  --epochs VALUE            Number of epochs (default: 50)"
            echo "  --lr VALUE                Learning rate (default: 0.0001)"
            echo "  --early_stopping VALUE    Patience for early stopping (default: 10)"
            echo "  --use_batch_correction    Apply batch correction"
            echo "  --use_global_features     Use global features from metadata"
            echo "  --device VALUE            Device to use (default: cuda)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required files exist
if [ ! -f "$METADATA_PATH" ]; then
    echo "Error: Metadata file not found: $METADATA_PATH"
    exit 1
fi

if [ ! -f "$CELL_DATA_PATH" ]; then
    echo "Error: Cell data file not found: $CELL_DATA_PATH"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVAL_DIR"

# Print settings
echo "======= Spatial-BERT Settings ======="
echo "Metadata file: $METADATA_PATH"
echo "Cell data file: $CELL_DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Evaluation directory: $EVAL_DIR"
echo "OS threshold: ${OS_THRESHOLD:-median (auto)}"
echo "k neighbors: $K_NEIGHBORS"
echo "Windows per sample: $WINDOWS_PER_SAMPLE"
echo "Hidden dimension: $HIDDEN_DIM"
echo "Number of heads: $NUM_HEADS"
echo "Number of layers: $NUM_LAYERS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Early stopping: $EARLY_STOPPING"
echo "Use batch correction: $USE_BATCH_CORRECTION"
echo "Use global features: $USE_GLOBAL_FEATURES"
echo "Device: $DEVICE"
echo "===================================="

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "Virtual environment activated!"

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "Warning: requirements.txt not found. Dependencies may not be installed."
else
    pip install -r requirements.txt
fi

# Build the command with optional parameters
TRAIN_CMD="python train.py --metadata_path $METADATA_PATH --cell_data_path $CELL_DATA_PATH --output_dir $OUTPUT_DIR"
TRAIN_CMD+=" --k_neighbors $K_NEIGHBORS --windows_per_sample $WINDOWS_PER_SAMPLE"
TRAIN_CMD+=" --hidden_dim $HIDDEN_DIM --num_heads $NUM_HEADS --num_layers $NUM_LAYERS"
TRAIN_CMD+=" --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LEARNING_RATE --early_stopping $EARLY_STOPPING"
TRAIN_CMD+=" --device $DEVICE"

# Add OS threshold if specified
if [ -n "$OS_THRESHOLD" ]; then
    TRAIN_CMD+=" --os_threshold $OS_THRESHOLD"
fi

# Add optional flags
if [ "$USE_BATCH_CORRECTION" = true ]; then
    TRAIN_CMD+=" --apply_batch_corr"
fi

if [ "$USE_GLOBAL_FEATURES" = true ]; then
    TRAIN_CMD+=" --use_global_features"
fi

# Run training
echo "Starting training..."
echo "Command: $TRAIN_CMD"
eval $TRAIN_CMD

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "Training completed successfully!"

# Build evaluation command
EVAL_CMD="python evaluate.py --metadata_path $METADATA_PATH --cell_data_path $CELL_DATA_PATH"
EVAL_CMD+=" --model_path $OUTPUT_DIR/spatial_bert_best.pt --output_dir $EVAL_DIR"
EVAL_CMD+=" --batch_size $BATCH_SIZE --device $DEVICE"

# Run evaluation
echo "Starting evaluation..."
echo "Command: $EVAL_CMD"
eval $EVAL_CMD

# Check if evaluation was successful
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi

echo "Evaluation completed successfully!"
echo "Training results are available in: $OUTPUT_DIR"
echo "Evaluation results are available in: $EVAL_DIR"

echo "All done!" 