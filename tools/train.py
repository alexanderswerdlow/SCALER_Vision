

import detectron2 as dt2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

import os


dt2.data.datasets.register_coco_instances("climb_dataset", {}, f"datasets/coco/annotations/instances_default.json", "datasets/coco/all_images")

cfg = dt2.config.get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (hold, volume, downclimb)
# cfg.MODEL.WEIGHTS = os.path.join(f"data_files/train.pth")
# cfg.MODEL.DEVICE = "cpu"
cfg.DATASETS.TRAIN = ("climb_dataset",)
cfg.DATASETS.TEST = ()
cfg.OUTPUT_DIR = 'data_files/training'
predictor = DefaultPredictor(cfg)
train_metadata = MetadataCatalog.get("climb_dataset")
DatasetCatalog.get("climb_dataset")


cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()