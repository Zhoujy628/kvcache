import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from palu.quant_utils import configure_latent_quantizer
import palu.model.svd_llama
#from vq_kvcache import VCache, K70Cache, K58Cache, K70K58VCache, K80K48VCache, K80K48VPPLCache
from transformers import DynamicCache
# from utils import load_config

import matplotlib.pyplot as plt



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="/home/jyzhou/OpenCompass/Palu2/Llama-3.1-8B_ratio-0.7_gs-4-fisher_uniform-whiten", required=False, help='Model to test')
    return parser.parse_args(args)

def get_ppl(device, data, max_length, model, past_key_values):
    input_ids = torch.tensor(
        data[max_length-1:max_length].astype(np.int64),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            labels=None,
            past_key_values=past_key_values,
            use_cache=True
        )
        # output = model(
        #     input_ids=input_ids,
        #     labels=None,
        #     past_key_values=None if past_key_values is None or len(past_key_values) == 0 else past_key_values,
        #     use_cache=True
        # )

    loss = torch.nn.functional.cross_entropy(
        output.logits[:, -1:, :].float().reshape(-1, output.logits.size(-1)),
        torch.tensor([int(data[max_length])], dtype=torch.long, device=device)
    )
    
    return loss

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name_or_path, device):
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    # model = model.eval()
    # return model, tokenizer
        # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

      # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)

    configure_latent_quantizer(
        model,
        n_bits=3,
        group_size=32,
        sym=True,
        clip_ratio=1.0,
        hadamard=True
    )

    model.eval()
    return model, tokenizer
    
    

def main():
    seed_everything(42)
    args = parse_args()
    model_name_or_path = args.model
    print(f"Testing ppl on model {model_name_or_path} (method)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.memmap('/home/yxwang/Dataset/kvcache/test.llama.bin', dtype=np.uint32, mode='r')

    losses = []
    max_len = []
    
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, device)
    past_key_values = DynamicCache()
    for length in range(1, 1000_0000):
        loss = get_ppl(device, data, length, model, past_key_values)
        # ppl = torch.exp(loss)
        print(f'length: {length}, loss: {loss.item()}')
        losses.append(loss.item())
        max_len.append(length)
    
    # draw loss curve with log max_length
    plt.plot(max_len, losses)
    plt.xscale('log')
    plt.show()
    plt.savefig('loss_llama3.1-8b_palu.png')
    plt.close()
    print(losses)

if __name__ == '__main__':
    main()
