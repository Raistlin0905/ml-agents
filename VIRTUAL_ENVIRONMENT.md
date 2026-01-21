## **Setting up the Virtual Environment**

### Prerequisites

> [!CAUTION]
> Python 3.10.12 (required)

Download Python 3.10.12 from [python.org](https://www.python.org/downloads/release/python-31012/)  
During installation, check "Add python.exe to PATH"  

### 1. Clone the Repository
```bash
git clone --branch fix-numpy-release-21-branch https://github.com/YourUsername/ml-agents.git
cd ml-agents
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Always activate the virtual environment after specifying the project path.
# Activate it (Windows Command Prompt) 
venv\Scripts\activate.bat

# For PowerShell or macOS/Linux, use:
# venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate    # macOS/Linux
```

### 3. Install Dependencies

#### For Mac users (Apple Silicon):
```bash
# Install grpcio pre-built wheel for Apple Silicon
pip install https://github.com/pietrodn/grpcio-mac-arm-build/releases/download/1.50.0/grpcio-1.50.0-cp310-cp310-macosx_11_0_arm64.whl
```

#### For all platforms:
```bash
# Install mlagents_envs first
pip install -e ./ml-agents-envs

# Then install the main package
pip install -e ./ml-agents
```

### 4. Verify Installation
```python
python -c "from mlagents_envs.environment import UnityEnvironment; print('Import successful!')"
```

# **Testing the connection: Run a training session**
This section guides you through the process of verifying that your Python environment can successfully run headless training.
### Prerequisites
- A built Unity environment executable (e.g., 3DBall).
- Your Python virtual environment (venv) is activated.

### 1. Start Headless Training
Open a terminal, navigate to your `ml-agents` project directory, and run the headless training script. This will start the training process without requiring the Unity Editor.

```bash
python headless_train.py 3DBall --no-graphics
``
```
### 2. Stop the Training
You can stop the training at any time by pressing `Ctrl+C` in your terminal.  
The Python process will finish its current step, save the training artifacts, and then exit.
```

**Visualizing the Training Metrics with TensorBoard**
### 1. Install or check that TensorBoard is in your virtual environment
```bash
pip install tensorboard
```
### 2. Run TensorBoard, pointing it to the results directory
```bash
tensorboard --logdir results
```
### 3. Open the given URL (usually `http://localhost:6006/`) in your web browser
### 4. Navigate to the "Scalars" tab to view graphs of key metrics