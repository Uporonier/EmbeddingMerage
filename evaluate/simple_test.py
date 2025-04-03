from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\desktop\merage\models\meraged\merged_llama_contrastive"  # 你的模型存放路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()

def ask_model(question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = "你知道山东大学吗"
print(ask_model(question))
