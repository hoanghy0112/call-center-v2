sudo apt install -y python3.10-venv
python3 -m venv .env
source .env/bin/activate
pip install -U "huggingface_hub[cli]"
huggingface-cli login