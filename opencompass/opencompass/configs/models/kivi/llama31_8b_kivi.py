from opencompass.models.llama3_1_8b_kivi import Llama3KIVI
import torch
models = [
    dict(
        type=Llama3KIVI,
        path='/data1/public/hf/meta-llama/Llama-3.1-8B',
        # running config
        abbr='llama-3.1-8b-kivi',
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== KIVI 配置 ====
        k_bits=4,    # 你学长说尽量压 80%，这里大概是 3bit
        v_bits=4 ,
        group_size=32,
        residual_length=32,
        torch_dtype="float16",
    )
]
