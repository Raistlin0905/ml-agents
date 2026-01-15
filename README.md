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
This section guides you through the process of verifying that your Python environment can successfully communicate with the Unity Editor to train an agent.
### Prerequisites
- Unity Editor is installed and opened.
- The ML-Agents package is installed in your Unity project.
- An example scene (e.g., 3DBall) is available in your project.
- Your Python virtual environment (venv) is activated.

### 1. Start the Python Training Process
Open a terminal, navigate to your `ml-agents/ml-agents` project directory, and run the learn.py module. This will start the process and wait for a connection from Unity.
```bash
python -m mlagents.trainers.learn
```
If this shows an error, you can either resume the training (keep the previous data) or force to overwrite previous data.
```bash
python -m mlagents.trainers.learn --resume
```
```bash
python -m mlagents.trainers.learn --force
```

**Expected Output**  
The command will start and hang at a message like:
```bash
[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.
```
### 2. Start the Unity Simulation
In the Unity Editor, open the example scene you want to train (e.g., `Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`).  
**Press the Play button in the Unity Editor.** This will start the simulation and connect to the Python script waiting on port 5004.

### 3. Look at the connection
Look back at your terminal. A successful connection will be confirmed with logs similar to these:
```bash
[INFO] Connected to Unity environment with package version 3.0.0-exp.1 and communication version 1.5.0
[INFO] Connected new brain: 3DBall?team=0
[WARNING] Behavior name 3DBall does not match any behaviors specified in the trainer configuration file. A default configuration will be used.
[INFO] Hyperparameters for behavior name 3DBall:
 trainer_type:   ppo
        hyperparameters:
          batch_size:   1024
          buffer_size:  10240
```
### 4. Stop the training
You can stop the training at any time by pressing the **Play** button again in the Unity Editor to stop the simulation.  
This will interrupt the Python process. It will finish its last step, save the training artifacts, and then exit.
```bash
[INFO] Learning was interrupted. Please wait while the graph is generated.
[INFO] Exported results\ppo\Pyramids\Pyramids-34048.onnx
[INFO] Copied results\ppo\Pyramids\Pyramids-34048.onnx to results\ppo\Pyramids.onnx.
[INFO] Exported results\ppo\3DBall\3DBall-20432.onnx
[INFO] Copied results\ppo\3DBall\3DBall-20432.onnx to results\ppo\3DBall.onnx.
```

# Checking the Results
After a successful run, the training outputs will be saved in the `./results` directory. Key files include:
- `.onnx` File: The trained neural network model (e.g., `results/ppo/3DBall.onnx`). This can be used in Unity for inference.
- `.pt` Files: PyTorch checkpoints (e.g., `checkpoint.pt`) that allow you to pause and resume training.
- `TensorBoard Log` Files: Data logs (e.g., `events.out.tfevents...`) used for visualizing training performance.

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


# Feature Importance with SHAP
SHAP (SHapley Additive exPlanations) is a model-agnostic method for explaining predictions of machine-learning models. It assigns each input feature a contribution value that reflects how much that feature influences the model’s output.

## Why we use SHAP?
At the data collection stage, we do not know in advance which hardware, configuration, or training metrics are most informative for predicting training runtime. Manually selecting features would risk introducing human bias and prematurely discarding useful information. SHAP can capture non-linear effects and feature interactions.

We therefore use SHAP to:
- Quantify the importance of each feature in predicting runtime
- Identify features that consistently contribute little or nothing to the prediction
- Reduce the feature space in a data-driven way before training our final models

In this project, SHAP is used only for feature weighting and selection, not as part of the final runtime prediction model.

## Data preparation
Before running SHAP, all features must be numeric. Thus, the dataset must be encoded using .... (see above) before giving it to SHAP.

## Install the required packages
Open terminal / PowerShell in the folder where the script is.
Run this:
```
python -m pip install pandas numpy shap xgboost
```

## How to use SHAP script
It trains a simple XGBoost regression model, computes SHAP feature importances, prints a ranking, and saves a reduced dataset with only the top-N most important features.

### 1. Change the CSV path to the target one

```
csv_path="ml-agents/mlagents/sample_runtime_data.csv",
```
Replace with your file, for example:
```
csv_path="data/3dball_joined_inner.csv",
```

### 2. Change the target column name (if needed)

```
target="runtime"
```
If your target column is called something else (e.g. duration_seconds), change it:
```
target="duration_seconds"
```

### 3. Change how many features you want to keep
```
clean_df = selector.select_features(top_n=4)
```
If you want the top 20 features:
```
clean_df = selector.select_features(top_n=20)
```

### 4. Change output filename (optional)
```
selector.save_cleaned_dataset(clean_df, "cleaned_training_data.csv")
```
Example:
```
selector.save_cleaned_dataset(clean_df, "cleaned_3dball.csv")
```

### 5. Run the script
In terminal:
```
python feature_importance.py
```




