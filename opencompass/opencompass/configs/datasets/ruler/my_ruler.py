from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k

ruler_datasets = []
ruler_datasets += ruler_datasets_4k
ruler_datasets += ruler_datasets_8k
ruler_datasets += ruler_datasets_16k
