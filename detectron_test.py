# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import torch
import torchvision
import argparse
import torchvision.transforms as transforms
from model.BVRRetina import BVRRetina
from coco_utils import _coco_remove_images_without_annotations
from itertools import compress
from eval import evaluate

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    # transforms.Resize((480, 640)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # transforms.Pad(32),
])

totensor = transforms.ToTensor()

valset = torchvision.datasets.CocoDetection('./datasets/coco/val2017',
    './datasets/coco/annotations/instances_val2017.json', transform = train_transform)
valset = _coco_remove_images_without_annotations(valset)
# valloader = torch.utils.data.DataLoader(valset,
#     batch_size=8,
#     shuffle=False,
#     collate_fn=collate_fn)

evaluator = COCOEvaluator("coco_2017_val", False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "coco_2017_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
