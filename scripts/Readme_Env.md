Detailed idea for setting up the Virtual Environment


conda create -n qwen_finetune
conda activate qwen_finetune


Install PyTorch & CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Install Necessary Libraries:
pip install unsloth transformers accelerate peft bitsandbytes datasets Pillow tqdm pandas scikit-learn

pip install qwen-vl-utils[decord]==0.0.8


Additionally you have to install Grok and Mistral for using their API
So make a separate venv for that

conda create -n grok_mistral_env
conda activate grok_mistral_env

pip install groq mistralai



After that you can start.