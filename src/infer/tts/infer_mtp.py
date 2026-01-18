from ...model.tts.modeling_mtp  import MedusaModelForQwen2_3Head
import time

device = "cuda:0"

# load model
model = MedusaModelForQwen2_3Head.from_pretrained(
    model_name_or_path="YOUR_PATH_TO_CKPT",
).to(device).eval()
print(model.num_parameters())

for name, param in model.named_parameters():
    if "speech_head" in name:
        print(name, param.numel())

tokenizer = model.tokenizer

prompt = tokenizer.bos_token + "[Human]: Read this sentence: They know it..\n [Assistant]: The speech is: "

start = time.time()
content = model.generate(
    **tokenizer([prompt], return_tensors="pt").to(device),
    use_cache=True, repetition_penalty=1.0
)
print(content)
end = time.time()
print(content['new_token'])
print("Time elapsed: ", end - start)
print(content['new_token'] / (end - start), 'tokens / s')
tokenizer = model.tokenizer