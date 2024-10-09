import os
import shutil
import io
from typing import List, Sequence
import numpy as np
import glob
import cv2

def main() -> None:
    input = 'data/lerf_ovs/waldo_kitchen/semantic'
    output = 'data/lerf_ovs/waldo_kitchen/semantic/colormap'
    os.makedirs(output, exist_ok=True)
    semantics = np.load(os.path.join(input, 'semantics.npy'))
    semantics = (semantics+1)/2
    colors=np.concatenate((semantics, np.zeros((1,3))))
    for image_name in glob.glob('[0-9][0-9][0-9][0-9][0-9].npy', root_dir=input):
        image = np.load(os.path.join(input, image_name))
        colormap=colors[image]
        cv2.imwrite(os.path.join(output, f'{os.path.splitext(image_name)[0]}.jpg'), colormap*255)

main()