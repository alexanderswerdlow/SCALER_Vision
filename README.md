## Installation Instructions

On Ubuntu 20.04:
```
pip install -r requirements.txt

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

sudo apt-get install -y libglib2.0-0 libgl1-mesa-glx
```

For other systems, you will need a different torch/detectron version.

To install librealsense, visit [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)


## Coordinate Frames

Default T265 (Viewing as if camera):
- X: Right, Y: Up, Z: Inwards
- Pitch: Up, Yaw: CCW, Roll: Left

Default D435 (Viewing as if camera):
- X: Right, Y: Down, Z: Out

Wall Frame (Viewing in front of wall):
- X: right, Y: up, Z: out
- Center of top left hole is 0,0

## Pipeline Overview

- Capture RGBD Images from D435
- Segment images using detectron2 and pre-trained 2d segmentation network
- Take each mask from segmentation and project that mask into a point cloud using camera intrinsics
- Filter point clouds by segmentation score, bounding box size, etc.
- Perform Ellipsoid Fitting on each point cloud