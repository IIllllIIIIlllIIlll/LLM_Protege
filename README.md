# LLM_Protege
LLM reverse teaching assistant

## Installation:

```
# Download and install Miniforge (an equivalent of Miniconda)
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge.sh
bash ~/miniforge.sh -b -p ~/miniforge3


# Activate base env and run init for the future
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
conda init

# Create conda environment
conda env create -f requirements.yaml

# Delete installer
rm ~/miniforge.sh
```

## Running:

**Before starting:**
- Please request the access from Mistral to their Mistral-7B-Instruct-v01 model on HuggigFace
- Please generate an access token from HuggingFace and store it in the `HUGGINGFACE_TOKEN` environment variable
- it is recommended to start a `tmux` session before launching the interface to allow gradio UI to persist after the terminal is closed


```
conda activate protege
python main_app.py
```

## Modifying:
You are welcome to change the system prompt directly in the `main_app.py` (line 25), along with examples of outputs (lines 179-180).
