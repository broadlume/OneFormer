from . import data  # register all new datasets
from . import modeling

# config
from .config import *

# dataset loading
from .data.dataset_mappers.coco_unified_new_baseline_dataset_mapper import COCOUnifiedNewBaselineDatasetMapper
from .data.dataset_mappers.oneformer_unified_dataset_mapper import (
    OneFormerUnifiedDatasetMapper,
)
from .data.dataset_mappers.semantic_oneformer_synthhomes_dataset_mapper import (
    SemanticOneFormerSynthHomesDatasetMapper,
)

# models
from .oneformer_model import OneFormer
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
