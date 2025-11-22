import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI

model_name = "/data1/public/hf/meta-llama/Llama-3.1-8B"  # 示例模型，实际你换成自己的 LLaMA 等
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KIVI 的 KV 量化参数（按你 wrapper 的默认/需求改）
K_BITS = 4      # 例如 K=2bit
V_BITS = 4      # 例如 V=2bit；若你想 4bit 就改 4
GROUP_SIZE = 32
RESIDUAL_LEN = 32

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM_KIVI.from_pretrained(model_name)
# model.to(device)
# model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- LlamaConfig + 注入 KIVI 参数（关键）---
config = LlamaConfig.from_pretrained(model_name)
config.k_bits = K_BITS
config.v_bits = V_BITS
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LEN
config.use_flash = True
config._flash_attn_2_enabled = True  # 避免 4D mask

# --- 构建 KIVI 模型 ---
model = LlamaForCausalLM_KIVI.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float16,  # 必须使用 fp16 或 bf16，FlashAttention 不支持 float32
    low_cpu_mem_usage=False,
    trust_remote_code=False,
)
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
    _ = model.generate(
        input_ids, 
        max_new_tokens=gen_max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024 ** 2)
    print(f"Peak GPU memory (allocated): {peak_mem_mb:.2f} MB")

# ===================================
# KV Cache Size
# ===================================
with torch.no_grad():
    out = model(input_ids, use_cache=True)

past_kv = out.past_key_values   # KIVI 的格式可能不同
print(f"num_layers in KV cache: {len(past_kv)}")

total_elements = 0
total_bytes = 0

# KIVI 的 past_key_values 格式：每层是长度为9的tuple，Item 1是K，Item 5是V
for layer_id, layer_cache in enumerate(past_kv):
    if isinstance(layer_cache, (tuple, list)):
        # 遍历所有元素并累加（KIVI 有多个元素，其中部分是 tensor）
        for item in layer_cache:
            if hasattr(item, 'numel') and hasattr(item, 'element_size'):
                layer_elems = item.numel()
                layer_bytes = layer_elems * item.element_size()
                total_elements += layer_elems
                total_bytes += layer_bytes

print(f"Total KV cache size: {total_bytes/1024/1024:.2f} MB")