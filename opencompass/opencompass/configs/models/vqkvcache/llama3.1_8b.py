from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3.1-8b',
        path='/fs-computility/plm/shared/wangyixuan/meta-llama/Llama-3.1-8B',
        max_out_len=32,
        batch_size=1,
        model_kwargs=dict(
            torch_dtype='torch.float'
        ), # for fairness
        run_cfg=dict(num_gpus=1),
    )
]
