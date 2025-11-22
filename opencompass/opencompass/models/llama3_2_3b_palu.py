mard: bool = True, # 使用 Hadamard 变换
                 lt_group_size: int = 32,  # 量化分组大小
                 lt_sym: bool = True,      # 对称量化
                 lt_clip_ratio: float = 1.0, # 裁剪比例
                 **kwargs):
        super().__init__(path=path, **kwargs)

        self.device = device
        self.path = path
        self.lt_bits = lt_bits
        self.lt_hadamard = lt_hadamard
        self.lt_group_size = lt_group_size
        self.lt_sym = lt_sym
        self.lt_clip_ratio = lt_clip_ratio

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, 
            use_fast=False,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 设置 torch_dtype
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == 'torch.float16':
                    self.torch_dtype = torch.float16
                elif torch_dtype == 'torch.float32':
                    self.torch_dtype = torch.float32
                else:
                    self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch_dtype
        else:
            self.torch_dtype = torch.float16

        # load model — Palu 自定义层会自动注入
        load_kwargs = dict(
            trust_remote_code=True, 
            device_map="auto",
            torch_dtype=self.torch_dtype
        )

        logger.info(f"Loading PALU Llama-3.2-3B model from {self.path} with dtype {self.torch_dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **load_kwargs)
        
        # === 按照官方方式设置量化参数 ===
        configure_latent_quantizer(
            self.model, 
            n_bits=self.lt_bits,
            group_size=self.lt_group_size,
            sym=self.lt_sym,
            clip_ratio=self.lt_clip_ratio,
            hadamard=self.lt_hadamard
        )
        
        logger.info(f"PALU quantization configured: lt_bits={self.lt_bits}, hadamard={self.lt_hadamard}, group_size={self.lt_group_size}")

        # 生成参数
        self.generation_kwargs = {
            'do_sample': False,
            'temperature': 1.0,
            'top_p': 1.0,
            'use_cache': True,
        }

    def generate(self, prompts: List[str], max_out_len=None, **kwargs) -> List[str]:
        max_new_tokens = max_out_len if max_out_len is not None else kwargs.pop("max_new_tokens", 512)

        # tokenizer 编码
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )

        # 获取模型设备
        if hasattr(self.model, "generation_device"):
            gen_device = self.model.generation_device
        else:
            gen_device = next(self.model.parameters()).device

        # 把输入送到设备
        inputs = {k: v.to(gen_device) for k, v in inputs.items()}

        # 合并生成参数：先使用传入的 kwargs，再应用模型的 generation_kwargs
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        
        # 确保 do_sample=False 并移除不兼容的参数
        generation_kwargs['do_sample'] = False
        incompatible_params = ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'typical_p']
        for param in incompatible_params:
            if param in generation_kwargs:
                logger.warning(f"Removing '{param}' parameter as it's incompatible with do_sample=False")
                generation_kwargs.pop(param)

        # 生成，使用清理后的参数
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )

        # decode 成字符串
        generated_texts = []
        for i, output in enumerate(outputs):
            # 只返回新生成的部分
            input_length = inputs['input_ids'][i].shape[0]
            generated = output[input_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)

        # step-5: 处理 stop words
        for stop in getattr(self, 'stop_words', []):
            generated_texts = [out.split(stop)[0] for out in generated_texts]

        return generated_texts

    def inference(self, data: dict, **kwargs) -> dict:
        """推理接口"""
        prompt = data["input"]
        result = self.generate([prompt], **kwargs)[0]
        return {"result": result}

    def get_token_len(self, prompt: str) -> int:
        """获取token长度"""from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
import sys
sys.path.append("/home/jyzhou/OpenCompass/Palu")

# 关键：强制触发 palullama 的注册逻辑
import palu.model.svd_llama
from palu.quant_utils import configure_latent_quantizer
import palu.model
import logging

logger = logging.getLogger(__name__)

@MODELS.register_module()
class PaluLlama32_3B(BaseModel):
    def __init__(self,
                 path: str,
                 device: str = "cuda",
                 torch_dtype=None,
                 # === 推理时量化参数 ===
                 lt_bits: int = 3,        # 3bit 量化
                 lt_hada
        return len(self.tokenizer.encode(prompt))
