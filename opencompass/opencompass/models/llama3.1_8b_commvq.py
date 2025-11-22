
from typing import Dict, List, Optional, Union
import importlib
import os

import torch
from mmengine.device import is_npu_available
from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# transformers 在 4.4x 之后引入了新的 Cache 接口，这里做兼容
try:
    from transformers.cache_utils import DynamicCache  # >=4.43 仍可用
except Exception:  # 兜底到老接口（极少数环境）
    DynamicCache = None

PromptType = Union[str, List[Dict[str, str]]]


def _get_possible_max_seq_len(max_seq_len: Optional[int], path: str) -> int:
    if max_seq_len is not None:
        return max_seq_len
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    for k in ("max_position_embeddings", "seq_length", "model_max_length"):
        if hasattr(cfg, k):
            return getattr(cfg, k)
    raise ValueError("max_seq_len is not provided and cannot be inferred from the model config.")


def _convert_base_messages(inputs: List[PromptType]) -> List[str]:
    """OpenCompass 的大多数长文任务都不是 chat 格式，这里把 PromptList 合并为平铺字符串。"""
    outs = []
    for _in in inputs:
        if isinstance(_in, str):
            outs.append(_in)
        else:
            pieces = []
            for item in _in:
                # item 形如 {'role': 'HUMAN'|'BOT'|'SYSTEM', 'prompt': '...'}
                # 长文任务通常只需要把所有 prompt 串起来
                pieces.append(item.get("prompt", ""))
            outs.append("".join(pieces))
    return outs


def _set_torch_dtype(model_kwargs: dict) -> dict:
    if "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        v = model_kwargs["torch_dtype"]
        mapping = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32 if hasattr(torch, "float32") else torch.float,
            "torch.float": torch.float,
            "auto": "auto",
            None: None,
        }
        if isinstance(v, str):
            model_kwargs["torch_dtype"] = mapping.get(v, torch.float16)
    return model_kwargs


def _maybe_move_to_device(tokens: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in tokens.items()}


def _build_commvq_cache(logger, commvq_cfg: dict):
    """尽力匹配官方 CommVQ 的缓存构造；如未安装则回退为 DynamicCache。

    兼容两类常见入口（不同版本可能存在差异）：
      - from commvq.cache import CommVQCache
      - from commvq import CommVQCache
    同时尝试 .from_pretrained(...) 与直接构造。
    """
    # 如果用户明确要求禁用（如做对比），直接返回普通 Cache
    if not commvq_cfg or commvq_cfg.get("disabled", False):
        logger.warning("CommVQ is disabled via config; using DynamicCache.")
        return DynamicCache() if DynamicCache is not None else None

    CacheCls = None
    for mod_name, cls_name in [
        ("commvq.cache", "CommVQCache"),
        ("commvq", "CommVQCache"),
    ]:
        try:
            mod = importlib.import_module(mod_name)
            CacheCls = getattr(mod, cls_name, None)
            if CacheCls is not None:
                break
        except Exception as e:
            continue

    if CacheCls is None:
        logger.warning("Cannot import CommVQCache from 'commvq'; falling back to DynamicCache. "
                       "Please install CommVQ by `pip install -e .` in its repo.")
        return DynamicCache() if DynamicCache is not None else None

    # 优先尝试 from_pretrained（常见于提供 codebook 的情形）
    try:
        if hasattr(CacheCls, "from_pretrained"):
            return CacheCls.from_pretrained(**{k: v for k, v in commvq_cfg.items() if v is not None})
    except TypeError:
        # 传参名不匹配时继续尝试直接构造
        pass
    except Exception as e:
        logger.warning(f"CommVQCache.from_pretrained failed with: {e}; will try direct __init__.")

    try:
        return CacheCls(**{k: v for k, v in commvq_cfg.items() if v is not None})
    except Exception as e:
        logger.warning(f"CommVQCache init failed with: {e}; falling back to DynamicCache.")
        return DynamicCache() if DynamicCache is not None else None


@MODELS.register_module()
class Llama3_1_8B_CommVQ(BaseModel):
    """Llama3.1-8B + CommVQ 的 OpenCompass 模型封装。

    设计目标：
      - 与 HuggingFaceBaseModel 的用法保持一致：给定 path、tokenizer、generation_kwargs 即可；
      - 额外提供 `commvq_cfg`（字典），用于指定 CommVQ 的 codebook/比特数 等参数；
      - 兼容 transformers 关于 KV cache 的旧（past_key_values）与新（cache）两种调用方式。

    典型 commvq_cfg（根据官方实现名可能存在差异，请按你本地版本调整）：
      commvq_cfg=dict(
          # 任选一种指路：codebook_dir 或 {value_codebook_path,key_codebook_path}
          codebook_dir='/path/to/commvq_llama3.1_8b_1bit',
          value_codebook_path='/path/to/value.ckpt',
          key_codebook_path='/path/to/key.ckpt',
          bitwidth=1,                # 1 或 2
          use_key=True,              # 是否量化K
          use_value=True,            # 是否量化V
          rope_commutative=True,     # 与 RoPE 对齐的可交换码本
      )
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 stop_words: Optional[List[str]] = None,
                 drop_middle: bool = False,
                 commvq_cfg: Optional[dict] = None,
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.template_parser = LMTemplateParser()
        self.tokenizer_only = tokenizer_only
        self.drop_middle = drop_middle
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self.stop_words = stop_words or []

        # tokenizer
        DEFAULT_TOK_KWARGS = dict(padding_side="left", truncation_side="left", trust_remote_code=True)
        tok_kwargs = {**DEFAULT_TOK_KWARGS, **(tokenizer_kwargs or {})}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or path, **tok_kwargs)

        # pad_token_id 兜底逻辑
        if pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        elif self.tokenizer.pad_token_id is None:
            try:
                gen_cfg = GenerationConfig.from_pretrained(path, trust_remote_code=True)
                if gen_cfg.pad_token_id is not None:
                    self.tokenizer.pad_token_id = gen_cfg.pad_token_id
                else:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            except Exception:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # model
        self.model = None
        if not tokenizer_only:
            DEFAULT_MODEL_KWARGS = dict(device_map="auto", trust_remote_code=True)
            model_kwargs = _set_torch_dtype({**DEFAULT_MODEL_KWARGS, **(model_kwargs or {})})
            if is_npu_available():
                model_kwargs["device_map"] = "npu"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            except ValueError:
                self.model = AutoModel.from_pretrained(path, **model_kwargs)

            # peft（可选）
            if peft_path is not None:
                from peft import PeftModel
                peft_kwargs = {**(peft_kwargs or {}), "is_trainable": False}
                self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

            self.model.eval()
            # 默认不采样，保持一致性
            try:
                self.model.generation_config.do_sample = False
            except Exception:
                pass

        self.generation_kwargs = generation_kwargs or {}
        self.commvq_cfg = commvq_cfg or {}

        # 把最关键的信息打到日志里，便于复现
        if self.commvq_cfg:
            self.logger.info(f"[CommVQ] cfg = {self.commvq_cfg}")
        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f"Unused argument {k}={v}")

    # --- OpenCompass 所需接口 ---
    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: Optional[List[str]] = None,
                 **kwargs) -> List[str]:
        assert self.model is not None, "tokenizer_only=True 无法进行生成"

        messages = _convert_base_messages(inputs)
        batch_size = len(messages)

        # tokenize：长文评测建议左 padding + 截断
        t_kwargs = dict(return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_seq_len)
        tokens = self.tokenizer.batch_encode_plus(messages, **t_kwargs)

        if self.drop_middle:
            # 可选：保留前后各半，丢中间（见你给的 vqkv 参考）
            input_ids = tokens["input_ids"]
            if input_ids.shape[-1] > self.max_seq_len:
                half = self.max_seq_len // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=-1)
            tokens = {"input_ids": input_ids}

        # 上设备
        tokens = _maybe_move_to_device(tokens, self.model.device)

        # 组装 generation kwargs
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs.update(kwargs)
        if max_out_len is not None:
            gen_kwargs["max_new_tokens"] = max_out_len
        if min_out_len is not None:
            gen_kwargs["min_new_tokens"] = min_out_len
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # stopping_criteria（字符串列表），转换为 multi-token 停止条件的逻辑可以按需加，
        # LongBench 默认不依赖它，这里不强制。
        _ = stopping_criteria  # 保留接口一致性

        # 构造 CommVQ 的 KV 缓存；若不可用则退化为 DynamicCache
        cache_obj = _build_commvq_cache(self.logger, self.commvq_cfg)

        # transformers 在 4.43+ 更推荐使用 `cache=`，老版本用 `past_key_values=`
        # 双路尝试，保证稳健
        try:
            outputs = self.model.generate(**tokens, **gen_kwargs, cache=cache_obj)
        except TypeError:
            outputs = self.model.generate(**tokens, **gen_kwargs, past_key_values=cache_obj)

        # 只取新生成部分
        if "input_ids" in tokens:
            prompt_len = tokens["input_ids"].shape[1]
            outputs = outputs[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def get_token_len(self, prompt: PromptType) -> int:
        msg = _convert_base_messages([prompt])[0]
        t = self.tokenizer(msg, add_special_tokens=True, return_tensors="pt")
        return int(t["input_ids"].shape[-1])

    # 可选：PPL相关接口（某些任务可能用得到）；不需要可不实现
    def get_ppl(self, inputs: List[PromptType], mask_length: Optional[List[int]] = None) -> List[float]:
        import torch.nn.functional as F
        pad_id = self.tokenizer.pad_token_id
        msgs = _convert_base_messages(inputs)
        toks = self.tokenizer.batch_encode_plus(
            msgs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True, max_length=self.max_seq_len
        )
        toks = _maybe_move_to_device(toks, self.model.device)
        logits = self.model(**toks)[0]
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = toks["input_ids"][:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_id,
            reduction="none",
        ).view(logits.size(0), -1)
        lens = (toks["input_ids"] != pad_id).sum(-1).float()
        ce = loss.sum(-1) / lens
        return ce.detach().cpu().numpy().tolist()
