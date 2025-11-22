from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import  Llama3KIVI
with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k

models = [
    dict(
        type=Llama3KIVI,
        # cache_type='llama-3.1-8b-kivi',
        # save_dir='./outputs/KIVI/checkpoints',
        # init=4,
        # local=1024,
        path='/data1/public/hf/meta-llama/Llama-3.1-8B',
        # model_kwargs=dict(
        #     torch_dtype='torch.float'
        # ),
        # running config
        abbr='llama-3.1-8b-kivi',
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== KIVI 配置 ====
        k_bits=4,    # 你学长说尽量压 80%，这里大概是 3bit
        v_bits=4,
        group_size=32,
        residual_length=32,
        torch_dtype="float16",
    )
]


work_dir = './outputs/KIVI/ruler_llama3.1/'

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

# python run.py eval/eval_on_ruler_kivi.py --dump-eval-details -r --max-num-workers ${YOUR_GPU_NUM}