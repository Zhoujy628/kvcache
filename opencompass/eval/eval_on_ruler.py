from opencompass.models import Llama3_1_8B_kvcache
from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k

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
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]

work_dir = './outputs/ruler'

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

# python run.py eval/eval_on_ruler.py --dump-eval-details -r --max-num-workers ${YOUR_GPU_NUM}