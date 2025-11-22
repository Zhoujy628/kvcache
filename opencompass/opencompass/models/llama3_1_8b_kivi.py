import torch
from transformers import LlamaConfig, AutoTokenizer
from opencompass.models.base import BaseModel
from opencompass.registry import MODELS

# 从 KIVI repo 导入
from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI


@MODELS.register_module()
class Llama3KIVI(BaseModel):
    """OpenCompass wrapper for KIVI LLaMA-3.2-3B"""
    is_api = False

    def __init__(self,
                 path: str,
                 device: str = "cuda",
                 k_bits: int = 4,
                 v_bits: int = 4,
                 group_size: int = 32,
                 residual_length: int = 32,
                 torch_dtype=torch.float16,
                 max_seq_len: int = 31500,
                 **kwargs):
        super().__init__(path=path, **kwargs)

        # 统一单卡设备（避免 auto 造成 CPU/GPU 混放）
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.stop_words = []

        # 兼容字符串 dtype
        if isinstance(torch_dtype, str):
            mapping = {
                'float16': torch.float16, 'fp16': torch.float16,
                'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
                'float32': torch.float32, 'fp32': torch.float32,
            }
            torch_dtype = mapping.get(torch_dtype.lower(), torch.float16)

        # --- 配置注入 KIVI 参数 ---
        config = LlamaConfig.from_pretrained(path)
        config.k_bits = k_bits
        config.v_bits = v_bits
        config.group_size = group_size
        config.residual_length = residual_length
        config.use_flash = True
        # 启用 2D/无 4D mask 路径，避免 _prepare_4d_causal_attention_mask 巨大膨胀
        config._flash_attn_2_enabled = True

        # --- 构建模型（强制单卡） ---
        self.model = LlamaForCausalLM_KIVI.from_pretrained(
            path,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        # 双重保险：确保使用非 4D mask 路径
        try:
            self.model.config._flash_attn_2_enabled = True
            self.model.model.config._flash_attn_2_enabled = True
        except Exception:
            pass

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _target_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return self.device

    def generate(self, prompts, max_out_len=None, min_out_len=None, stopping_criteria=[], **kwargs):
        # --- step 1: tokenize ---
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=False,                 # 关键：不 padding，避免 4D mask
            truncation=True,
            return_attention_mask=False    # 不返回 attention_mask
        )

        # 移除可能残留的 attention_mask
        inputs.pop("attention_mask", None)

        # 统一设备到嵌入所在设备，避免 CPU/CUDA 混放
        dev = self._target_device()
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(dev, non_blocking=True)

        # --- step 2: generation kwargs ---
        generation_kwargs = getattr(self, "generation_kwargs", {}).copy()
        generation_kwargs.update(kwargs)
        if max_out_len is not None:
            generation_kwargs["max_new_tokens"] = max_out_len
        if min_out_len is not None:
            generation_kwargs["min_new_tokens"] = min_out_len
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # --- step 3: forward ---
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs["input_ids"], **generation_kwargs)

        # --- step 4: decode - 只返回新生成的部分 ---
        generated_texts = []
        input_ids = inputs["input_ids"]
        for i, output in enumerate(outputs):
            in_len = input_ids[i].shape[0]
            generated_tokens = output[in_len:]
            generated_texts.append(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))

        # --- step 5: process stop word ---
        for stop in stopping_criteria + self.stop_words:
            generated_texts = [out.split(stop)[0] for out in generated_texts]

        return generated_texts

    def inference(self, data: dict, **kwargs):
        prompt = data["input"]
        result = self.generate([prompt], **kwargs)[0]  # 修正：不再取 [0][0]
        return {"result": result}

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))