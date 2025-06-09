# 3D-Scene-In-Out-Painting

A comprehensive project for 3D scene inpainting and outpainting using advanced machine learning techniques.

## Setup Instructions

### Prerequisites
- Python 3.10
- Git
- CUDA-compatible GPU (recommended for optimal performance)

### Environment Setup

#### Option 1: Using Conda (Recommended)

1. **Create a new conda environment with Python 3.10:**
   ```bash
   conda create -n 3D-Scene-In-Out-Painting python=3.10
   ```

2. **Activate the environment:**
   ```bash
   conda activate 3D-Scene-In-Out-Painting
   ```

3. **Navigate to the project directory:**
   ```bash
   cd 3D-Scene-In-Out-Painting
   ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Option 2: Using Python venv

1. **Create a virtual environment:**
   ```bash
   python3.10 -m venv 3D-Scene-In-Out-Painting
   ```

2. **Activate the environment:**
   - On Linux/macOS:
     ```bash
     source 3D-Scene-In-Out-Painting/bin/activate
     ```
   - On Windows:
     ```bash
     3D-Scene-In-Out-Painting\Scripts\activate
     ```

3. **Navigate to the project directory:**
   ```bash
   cd 3D-Scene-In-Out-Painting
   ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Verification

After installation, verify that the environment is set up correctly:

```bash
python --version  # Should show Python 3.10.x
pip list | grep torch  # Should show PyTorch installation
```

### Usage

Once the environment is set up and activated, you can run the project applications:

```bash
# Example: Run the GUI application
bash Inpaint-Anything/script/run_gui.sh
```

### Troubleshooting

- **CUDA Issues**: If you encounter CUDA-related errors, ensure you have compatible NVIDIA drivers installed
- **Memory Issues**: For large models, ensure you have sufficient RAM and GPU memory
- **Dependencies**: If specific packages fail to install, try installing them individually with pip

### Project Structure

- `Inpaint-Anything/` - Main inpainting functionality and GUI applications
- `requirements.txt` - Python dependencies
- `script/` - Utility scripts for running applications

### Notes

- This project requires significant computational resources for optimal performance
- GPU acceleration is highly recommended
- Some models may require additional downloads during first use 