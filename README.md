
On Ubuntu 20.04:
```
pip install -r requirements.txt

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

sudo apt-get install -y libglib2.0-0 libgl1-mesa-glx
```

To install librealsense, visit [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)