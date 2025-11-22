from opencompass.models import Llama3_1_8B_kvcache
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.needlebench.my_needlebench.needlebench import needlebench_origin_en_datasets
    from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        type=Llama3_1_8B_kvcache,
        cache_type='k80k48vcache',
        save_dir='/fs-computility/plm/shared/wangyixuan/vqkvcache/LongBench-v2/LongBench/exp/0827_longbench_K80K48VCache',
        init=4,
        local=1024,
        path='/fs-computility/plm/shared/wangyixuan/meta-llama/Llama-3.1-8B',
        model_kwargs=dict(
            torch_dtype='torch.float'
        ),
        # running config
        abbr='llama-3.1-8b-k80k48vcache',
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './outputs/niah'

infer = dict(
    partitioner=dict(type=NaivePartitioner), 
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

# python run.py eval/eval_on_niah.py --dump-eval-details -r --max-num-workers ${YOUR_GPU_NUM}