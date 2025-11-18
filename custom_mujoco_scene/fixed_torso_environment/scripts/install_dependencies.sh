#!/bin/bash
# Install all dependencies for G1 Reaching Task Training
# Includes MuJoCo, Stable-Baselines3, and optional packages

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Navigate to parent directory (fixed_torso_environment)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================="
echo "Installing Dependencies for G1 Training"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "[OK] Conda found"
echo ""

# Get current environment name
CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "Current conda environment: $CONDA_ENV"
echo ""

# Confirm installation
read -p "Install dependencies in '$CONDA_ENV'? [Y/n]: " confirm
confirm=${confirm:-Y}

if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "========================================="
echo "Installing packages..."
echo "========================================="
echo ""

# Install core dependencies with conda
echo "[1/4] Installing core packages with conda..."
conda install -y -c conda-forge \
    numpy \
    matplotlib \
    scipy

echo ""
echo "[2/4] Installing MuJoCo..."
pip install -q mujoco

echo ""
echo "[3/4] Installing Stable-Baselines3 and Gymnasium..."
pip install -q stable-baselines3[extra]
pip install -q gymnasium
pip install -q tensorboard

echo ""
echo "[4/4] Installing optional packages for GPU-accelerated training..."

# Install Brax for GPU-accelerated training (optional)
read -p "Install Brax (GPU-accelerated training)? [y/N]: " install_brax
install_brax=${install_brax:-N}

if [[ $install_brax =~ ^[Yy]$ ]]; then
    echo "Installing JAX and Brax..."
    pip install -q "jax[cpu]" 2>/dev/null || pip install -q jax
    pip install -q brax==0.10.5
    echo "[OK] Brax installed"
else
    echo "[SKIP] Skipping Brax installation"
fi

echo ""
echo "========================================="
echo "Verifying Installation"
echo "========================================="
echo ""

# Verify core packages
python << 'EOF'
import sys

packages = {
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'mujoco': 'MuJoCo',
    'stable_baselines3': 'Stable-Baselines3',
    'gymnasium': 'Gymnasium',
    'tensorboard': 'TensorBoard',
}

optional_packages = {
    'jax': 'JAX',
    'brax': 'Brax',
}

print("Core packages:")
all_good = True
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {name:25s} {version}")
    except ImportError:
        print(f"  [FAIL] {name:25s} NOT INSTALLED")
        all_good = False

print("\nOptional packages:")
for module, name in optional_packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  [OK] {name:25s} {version}")
    except ImportError:
        print(f"  [SKIP] {name:25s} not installed (optional)")

if all_good:
    print("\n[SUCCESS] All core packages installed successfully!")
    sys.exit(0)
else:
    print("\n[ERROR] Some core packages failed to install")
    sys.exit(1)
EOF

VERIFY_STATUS=$?

echo ""
echo "========================================="

if [ $VERIFY_STATUS -eq 0 ]; then
    echo "[SUCCESS] Installation Complete!"
    echo "========================================="
    echo ""
    echo "Training options available:"
    echo "  ./scripts/train_sb3.sh           (Standard SB3 training)"
    echo "  python train_sb3_improved.py     (Improved config, 500k steps)"
    echo "  python train_with_brax.py        (GPU-accelerated, if Brax installed)"
    echo ""
    echo "Visualization:"
    echo "  ./scripts/visualize.sh           (View trained models)"
    echo ""
else
    echo "[WARNING] Installation completed with warnings"
    echo "========================================="
    echo ""
    echo "Some packages may need manual installation."
    echo "Try running: pip install matplotlib numpy mujoco stable-baselines3"
    echo ""
fi
