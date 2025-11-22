from opencompass.models import Llama3_1_8B_kvcache
# from opencompass.partitioners import SizePartitioner, NaivePartitioner, NumWorkerPartitioner
# from opencompass.runners import LocalRunner
# from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

# infer = dict(
#     runner=dict(
#         max_num_workers=2,
#     )
# )

models = [
    dict(
        type=Llama3_1_8B_kvcache,
        cache_type='vcache',
        save_dir='/fs-computility/plm/shared/wangyixuan/vqkvcache/LongBench-v2/LongBench/exp/0809_vcache_32_512',
        init=4,
        local=1024,
        path='/fs-computility/plm/shared/wangyixuan/meta-llama/Llama-3.1-8B',
        model_kwargs=dict(
            torch_dtype='torch.float'
        ),
        # running config
        abbr='llama-3.1-8b-vcache',
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
