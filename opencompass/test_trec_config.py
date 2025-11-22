from mmengine.config import read_base

with read_base():
    from .configs.datasets.LongBench.LongBench_trec.LongBench_trec_gen_9adeeb import LongBench_trec_datasets
    from .configs.models.kivi.llama32_3b_kivi import models

datasets = LongBench_trec_datasets
work_dir = './outputs/test_trec'
