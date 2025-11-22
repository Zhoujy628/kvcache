from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models.llama3_2_3b_palu import PaluLlama32_3B 
import torch

with read_base():
    from opencompass.configs.datasets.needlebench.my_needlebench.needlebench import needlebench_origin_en_datasets
from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        type=PaluLlama32_3B,
        path='/home/jyzhou/OpenCompass/Palu/Llama-3.2-3B_ratio-0.7_gs-4-fisher_uniform-whiten',
        abbr='llama-3.2-3b-palu',
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== PALU 修复后的配置 ====
        max_seq_len=32768,
        lt_bits=3,
        lt_hadamard=True,
        lt_group_size=32,   # 改回32，或者尝试其他能整除的值如16, 8
        lt_sym=True,
        lt_clip_ratio=1.0,
        torch_dtype="float16",
    )
]

work_dir = './outputs/Palu_3/niah/llama3.2'

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

# python run.py eval/eval_on_niah_palu_llama3.2.py --dump-eval-details -r --max-num-workers ${YOUR_GPU_NUM}