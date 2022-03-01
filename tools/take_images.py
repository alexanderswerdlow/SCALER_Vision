import d435_sub
from PIL import Image
  
i = 0
while True:
    input(f"Press enter to take image {i}")
    (color_image, depth_image, ir_image), (intrinsics, extrinsic) = d435_sub.get_rgbd()
    Image.fromarray(color_image).save('data_files/charuco/{i}.png')