git clone https://github.com/r-buitrago/LLaVA /code/LLaVA

pip install aeon
pip install openai
pip install wandb
cd /code/LLaVA

pip install --upgrade pip

# !pip install deepspeed
# !pip install flash-attn
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation


