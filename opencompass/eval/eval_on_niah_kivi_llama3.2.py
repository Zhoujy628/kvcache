from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models.llama3_2_3b_kivi import Llama32_3B_KIVI
import torch
with read_base():
    from opencompass.configs.datasets.needlebench.my_needlebench.needlebench import needlebench_origin_en_datasets

# 在 read_base() 外面导入 summarizer
from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        type=Llama32_3B_KIVI,
        path='/data1/public/hf/meta-llama/Llama-3.2-3B',
        abbr='llama-3.2-3b-kivi',
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== KIVI 配置 ====
        k_bits=4,    
        v_bits=4,
        group_size=32,
        residual_length=32,
        torch_dtype="float16",
    )
]

work_dir = './outputs/KIVI/niah_llama3.2'

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

# python run.py eval/eval_on_niah_palu_llama3.1.py --dump-eval-details -r --max-num-workers ${YOUR_GPU_NUM}