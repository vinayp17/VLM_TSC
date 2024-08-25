llava_root=$1
git clone https://github.com/r-buitrago/LLaVA llava_root

pip install aeon
pip install openai
pip install wandb
cd $llava_root

pip install --upgrade pip

pip install PyWavelets

# !pip install deepspeed
# !pip install flash-attn
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation


