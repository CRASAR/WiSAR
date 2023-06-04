conda create --name WiSAR python=3.9 -y
conda activate WiSAR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt