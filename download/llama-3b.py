import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.system('huggingface-cli download --resume-download meta-llama/Llama-3.2-3B --local-dir D:\\desktop\\merage\\models\\llama-3b')
# meta-llama/Llama-3.2-3B