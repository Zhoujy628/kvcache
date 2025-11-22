from opencompass.models.llama3_1_8b_palu import PaluLlama3

models = [
    dict(
        type='PaluLlama3',
        abbr='llama-3.1-8b-palu',
        path='/home/jyzhou/OpenCompass/Palu2/Llama-3.1-8B_ratio-0.7_gs-4-fisher_uniform-whiten',
        max_out_len=512,      # 减小输出长度以节省显存
        batch_size=1,
        # === PALU 推理量化参数 ===
        lt_bits=3,            # 3bit 量化，达到总压缩率 86%
        lt_hadamard=True,     # 使用 Hadamard 变换
        lt_group_size=32,     # 量化分组大小
        lt_sym=True,          # 对称量化
        lt_clip_ratio=1.0,    # 裁剪比例
        # === 明确的生成参数配置 ===
        generation_kwargs=dict(
            do_sample=False,  # 确保确定性生成
            # 不设置 temperature, top_p 等与 do_sample=False 冲突的参数
        ),
        model_kwargs=dict(
            torch_dtype='torch.float16'  # 使用 float16 节省显存
        ),
        run_cfg=dict(num_gpus=1),
    )
]
