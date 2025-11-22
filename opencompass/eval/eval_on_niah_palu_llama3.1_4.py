from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models.llama3_2_3b_palu import PaluLlama32_3B

with read_base():
    from opencompass.configs.datasets.needlebench.my_needlebench.needlebench import needlebench_origin_en_datasets

# 在 read_base() 外面导入 summarizer
from opencompass.configs.summarizers.needlebench import needlebench_summarizer as summarizer

datasets = []
datasets += needlebench_origin_en_datasets

models = [
    dict(
        type=PaluLlama32_3B,
        path='/home/jyzhou/OpenCompass/Palu/Llama-3.1-8B_ratio-0.7_gs-4-fisher_uniform-whiten',
        abbr='llama-3.1-3b-palu-4',
        max_out_len=32,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        # ==== PALU 优化后的配置 ====
        lt_bits=4,          # 使用4bit而不是3bit，减少量化损失
        lt_hadamard=True,   # 保持Hadamard变换
        lt_group_size=32,   # 增大分组大小，减少量化损失
        lt_sym=True,        # 对称量化
        lt_clip_ratio=1.0,  # 裁剪比例
        torch_dtype="float16",
    )
]

work_dir = './outputs/PALU_4/niah_llama3.1'

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
