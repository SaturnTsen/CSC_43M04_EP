from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2TokenizerFast
from transformers import set_seed, TextStreamer
set_seed(42)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(model_name)

input_text = "Q: Translate into English 'les voitures de la Commission européenne sont vertes' A:"
inputs = tokenizer(input_text, return_tensors="pt")

streamer = TextStreamer(tokenizer)

model.eval()
model.generate(**inputs,
               max_length=100,
               do_sample=False,
               temperature=None,
               top_p=None,
               top_k=None,
               streamer=streamer)