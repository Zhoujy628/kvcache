import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import DynamicCache
import matplotlib.pyplot as plt

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/data1/public/hf/meta-llama/Llama-3.1-8B", required=False, help='Model to test')
    return parser.parse_args(args)

def get_ppl(device, data, max_length, model, past_key_values):
    input_ids = torch.tensor(
        data[max_length-1:max_length].astype(np.int64),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        if past_key_values is None:
            output = model(
                input_ids=input_ids,
                use_cache=True
            )
        else:
            output = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
    
    loss = torch.nn.functional.cross_entropy(
        output.logits[:, -1:, :].float().reshape(-1, output.logits.size(-1)),
        torch.tensor([int(data[max_length])], dtype=torch.long, device=device)
    )
    
    return loss, output.past_key_values

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_pretrained(model_name_or_path)
    config.k_bits = 4
    config.v_bits = 4
    config.group_size = 32
    config.residual_length = 32
    config.use_flash = True
    config._flash_attn_2_enabled = True

    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return model, tokenizer

def main():
    seed_everything(42)
    args = parse_args()
    model_name_or_path = args.model
    print(f"Testing ppl on model {model_name_or_path} (KIVI baseline)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.memmap('/home/yxwang/Dataset/kvcache/test.llama.bin', dtype=np.uint32, mode='r')

    losses = []
    max_len = []

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    past_key_values = None

                
    for length in range(1, 1000_0000):
        loss, past_key_values = get_ppl(device, data, length, model, past_key_values)

        print(f'length: {length}, loss: {loss.item()}')

        losses.append(loss.item())
        max_len.append(length)

    # 保存结果
    plt.plot(max_len, losses)
    plt.xscale('log')
    plt.show()
    plt.savefig('kivi_loss_llama3.1-8b.png')
    plt.close()
    print(losses)
if __name__ == '__main__':
    main()