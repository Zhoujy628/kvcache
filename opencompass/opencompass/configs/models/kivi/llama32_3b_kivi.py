from opencompass.models.llama3_2_3b_kivi import Llama32_3B_KIVI

models = [
    dict(
        type=Llama32_3B_KIVI,
        path='/data1/public/hf/meta-llama/Llama-3.2-3B',
        # running config
        abbr='llama-3.2-3b-kivi',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== KIVI 配置 ====
        k_bits=4,             # 4bit KV Cache 量化
        v_bits=4,
        group_size=32,
        residual_length=32,
        torch_dtype="float16",
    )
]
