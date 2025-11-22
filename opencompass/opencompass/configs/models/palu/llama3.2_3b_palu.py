from opencompass.models.llama3_1_8b_palu import PaluLlama3

models = [
    dict(
        type='PaluLlama3',
        path='/home/jyzhou/OpenCompass/Palu2/Llama-3.2-3B_ratio-0.7_gs-4-fisher_uniform-whiten',
        device='cuda',
        abbr='llama-3.2-3b-palu',
        max_out_len=31500,
        batch_size=1,
        lt_bits=3,        # 3bit 量化，达到总压缩率 86%
        lt_hadamard=True,
        run_cfg=dict(num_gpus=2),
    )
]