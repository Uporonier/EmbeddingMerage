import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.system('huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir D:\\desktop\\merage\\models\\llama-1b')
# meta-llama/Llama-3.2-3B