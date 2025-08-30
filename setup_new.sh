# 0) Use the SAME venv you train with
	source /venv/main/bin/activate
	which python; which deepspeed

	# 1) Make sure CUDA 11.7 toolkit is active
	export CUDA_HOME=/usr/local/cuda-11.7
	export PATH="$CUDA_HOME/bin:$PATH"
	export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib64"
	nvcc -V  # should show release 11.7

	# 2) Keep NumPy in the 1.x line (avoids ABI issues with older wheels)
	pip install --upgrade --force-reinstall "numpy==1.26.4"

	# 3) Install Torch 2.0.1 **cu117** (matches your toolkit)
	pip uninstall -y torch torchvision torchaudio || true
	pip install --index-url https://download.pytorch.org/whl/cu117 \
		  "torch==2.0.1" "torchvision==0.15.2" "torchaudio==2.0.2"

	python - <<'PY'
import torch; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
# Expect: 2.0.1 / 11.7

# 4) Tooling for building flash-attn
pip install -U ninja "pybind11>=2.12" packaging setuptools wheel

# 5) Remove any mismatched flash-attn artifacts
pip uninstall -y flash-attn flash_attn || true
find /venv/main/lib/python3.10/site-packages -maxdepth 1 -name 'flash_attn*' -print -exec rm -rf {} +

# 6) Build a flash-attn version compatible with Torch 2.0.x
# Start with 2.0.8 (good with Torch 2.0). If it complains, try 2.1.0 next.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-70;75;80;86;89}"  # adjust if you know your GPU SM
pip install --no-build-isolation --no-binary=:all: "flash-attn==2.0.8"

# (fallback if needed)
# pip install --no-build-isolation --no-binary=:all: "flash-attn==2.1.0"

# 7) Sanity test
python - <<'PY'
import torch
print("Torch OK:", torch.__version__, "CUDA", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
print("flash-attn import OK")
PY

