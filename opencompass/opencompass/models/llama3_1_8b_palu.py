from typing import List
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
class PaluLlama3(BaseModel):
    def __init__(self,
                 path: str,
                 device: str = "cuda",
                 torch_dtype=None,
                 # === 推理时量化参数 ===
                 lt_bits: int = 3,        # 3bit 量化
                 lt_hadamard: bool = True, # 使用 Hadamard 变换
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
            padding_side='left', 
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

        logger.info(f"Loading PALU model from {self.path} with dtype {self.torch_dtype}")
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

    def generate(self, prompts: List[str], max_out_len=None, **kwargs) -> List[str]:
        max_new_tokens = max_out_len if max_out_len is not None else kwargs.pop("max_new_tokens", 512)

        # tokenizer 编码
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
        )

        # 确保输入在正确的设备上
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 合并生成参数，移除冲突参数
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        
        # 移除与 do_sample=False 冲突的参数
        conflicting_params = ['temperature', 'top_p', 'top_k', 'repetition_penalty']
        for param in conflicting_params:
            if param in generation_kwargs and not generation_kwargs.get('do_sample', True):
                generation_kwargs.pop(param)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # 确定性生成
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # 返回空结果而不是崩溃
            return [""] * len(prompts)

        # decode 成字符串
        generated_texts = []
        for i, output in enumerate(outputs):
            # 只返回新生成的部分
            input_length = inputs['input_ids'][i].shape[0]
            generated = output[input_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def inference(self, data: dict, **kwargs) -> dict:
        prompt = data["input"]
        result = self.generate([prompt], **kwargs)[0]
        return {"result": result}

    def _move_to_first_device(self):
        """移动到第一个设备（与 OpenCompass 兼容）"""
        return self.model