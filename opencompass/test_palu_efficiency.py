import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys #Palu 自定义模import sys
sys.path.insert(0, '/home/jyzhou/OpenCompass/Palu2')
from palu.model.svd_llama import PaluLlamaConfig,PaluLlamaForCausalLM

model_name = "/home/jyzhou/OpenCompass/Palu2/Llama-3.1-8B_ratio-0.7_gs-4-fisher_uniform-whiten"  # 示例模型，实际你换成自己的 LLaMA 等
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

# 构造一个示例输入
# TODO, not use customized input, use pg-19
text = "This is a test input for efficiency measurement."
inputs = tokenizer(text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]
print(f"Prompt Token's Number: {input_ids.shape[-1]}")

# ===================================
# Prefill/Decode Latency + tokens/s
# ===================================
import time

# 生成参数：例如生成 50 个新 token
gen_max_new_tokens = 50

# 预热一次，防止冷启动影响
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=5)

torch.cuda.synchronize()  # 确保 GPU 队列清空

# 正式计时
start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=gen_max_new_tokens,
        do_sample=False,
        use_cache=True,      # 一般默认就开了
    )
torch.cuda.synchronize()
end = time.perf_counter()

total_time = end - start
total_new_tokens = outputs.shape[1] - input_ids.shape[1]

print(f"Total time: {total_time:.4f} s")
print(f"Generated new tokens: {total_new_tokens}")
print(f"End-to-end tokens/s: {total_new_tokens / total_time:.2f} tokens/s")

# ===================================
# prefill latency & decode latency
# Prefill：一次性处理完整上下文
# ===================================
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    out = model(input_ids, use_cache=True)
torch.cuda.synchronize()
t1 = time.perf_counter()

past_kv = out.past_key_values  # 里面就是 KV cache
prefill_time = t1 - t0
prefill_tokens = input_ids.shape[1]
print(f"Prefill time: {prefill_time:.4f}s, tokens: {prefill_tokens}, "
      f"prefill TPS: {prefill_tokens / prefill_time:.2f}")

# Decode：逐 token 生成 N 个新 token
decode_steps = 30
last_token = input_ids[:, -1:]

torch.cuda.synchronize()
t2 = time.perf_counter()
with torch.no_grad():
    for _ in range(decode_steps):
        out = model(last_token, use_cache=True, past_key_values=past_kv)
        logits = out.logits
        past_kv = out.past_key_values
        # 选一个 next token（这里 greedy）
        last_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
torch.cuda.synchronize()
t3 = time.perf_counter()

decode_time = t3 - t2
print(f"Decode time: {decode_time:.4f}s, steps: {decode_steps}, "
      f"decode TPS: {decode_steps / decode_time:.2f}")

# ===================================
# GPU peak memory
# ===================================
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=gen_max_new_tokens)

if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024 ** 2)
    print(f"Peak GPU memory (allocated): {peak_mem_mb:.2f} MB")

# ===================================
# KV Cache Size
# ===================================
with torch.no_grad():
    out = model(input_ids, use_cache=True)

past_kv = out.past_key_values   # list, 长度 = num_layers
print(f"num_layers in KV cache: {len(past_kv)}")

total_elements = 0
total_bytes = 0
for layer_id, (k, v) in enumerate(past_kv):
    # k, v 形状一般是 [batch, num_heads, seq_len, head_dim]
    layer_elems = k.numel() + v.numel()
    layer_bytes = layer_elems * k.element_size()
    total_elements += layer_elems
    total_bytes += layer_bytes
    # print(f"Layer {layer_id}: shape K={tuple(k.shape)}, V={tuple(v.shape)}, "
    #       f"size={layer_bytes/1024/1024:.2f} MB")

print(f"Total KV cache size: {total_bytes/1024/1024:.2f} MB")