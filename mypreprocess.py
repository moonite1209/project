import os
import gc
import logging
import random
import shutil
import io
import argparse
from typing import List, Sequence
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torch
import torchvision
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from tqdm import tqdm

image_path = None
save_path = None
mask_generator = None
predictor = None
state = None
class Segments:
    smaps: list
    def __init__(self, image_num, image_height, image_width) -> None:
        self.image_num = image_num
        self.image_height = image_height
        self.image_width = image_width
        self.cursor = 0
        self.smaps = [torch.full((image_height, image_width), -1, device='cuda') for i in range(image_num)]
        
    def remove_duplicate(self, frame_idx, object_ids, masks, prompt: list):
        smap = self.smaps[frame_idx]
        for i, mask in enumerate(masks):
            if duplicate(smap, mask)>0.8:
                prompt[i]=None
        return [p for p in prompt if p!=None]
    
    def add_masks(self, frame_idx, object_ids, masks):
        smap=self.smaps[frame_idx]
        for id, mask in zip(object_ids, masks, strict=True):
            smap[mask>0] = id

class Entities:
    container: list
    def __init__(self, iamge_num) -> None:
        self.container = []

    def add_entities(self, current_frame, ids: list, masks, prompt):
        object_ids = []
        for i, mask in zip(ids, masks, strict=True):
            object_ids.append(len(self.container))
            self.container.append({
                'prompt_frame': current_frame,
                'prompt': prompt[i],
                'mask': mask
            })
        return object_ids

    def get_colormap(self):
        colormap=[torch.rand(3) for i in range(len(self.container))]
        colormap.append(torch.zeros(3))
        return torch.stack(colormap).cuda()


def duplicate(smap, mask):
    smap=smap>=0
    mask=mask>0
    return (smap&mask).sum()/mask.sum()

def iou(mask1, mask2):
    mask1=mask1>=0
    mask2=mask2>=0
    return (mask1 & mask2).sum()/(mask1 | mask2).sum()

def save_smap(segments: Segments, entities: Entities):
    colormap = entities.get_colormap()
    for i, smap in enumerate(segments.smaps):
        fmap=colormap[smap]
        torchvision.utils.save_image(fmap.permute(2,0,1), os.path.join(save_path, f'{str(i).rjust(5,'0')}.jpg'))

def get_entities(frame_idx, prompt):
    global image_path, predictor, state
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # state = predictor.init_state(image_path)
        predictor.reset_state(state)
        for id, p in enumerate(prompt):
            predictor.add_new_mask(state, frame_idx, id, p)
        for frame_index, object_ids, masks in predictor.propagate_in_video(state): # masks: (n, 1, h, w)
            masks = masks.squeeze(1)
            if frame_index == frame_idx:
                break
    return frame_index, object_ids, masks

def prompt_filter(mask):
    mask=mask['segmentation']>0
    x = np.zeros_like(mask)
    x[0, ...] = True
    x[-1, ...] = True
    x[..., 0] = True
    x[..., -1] = True
    return ~np.any(mask & x)

def get_prompt(image: torch.Tensor):
    global mask_generator
    masks=mask_generator.generate(image.numpy())
    return [torch.from_numpy(mask['segmentation']) for mask in masks if prompt_filter(mask)]


def video_segment(images: np.ndarray):
    global image_path, mask_generator, predictor, state
    segments = Segments(len(images), images.shape[1], images.shape[2])
    entities  =Entities(len(images))
    for current_frame, image in tqdm(enumerate(images), desc='video_segment'):
        prompt = get_prompt(image)
        frame_idx, object_ids, masks = get_entities(current_frame, prompt)
        prompt = segments.remove_duplicate(frame_idx, object_ids, masks, prompt)
        if len(prompt)==0:
            continue
        ids = entities.add_entities(current_frame, object_ids, masks, prompt)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # state = predictor.init_state(image_path)
            predictor.reset_state(state)
            for id, p in zip(ids, prompt, strict=True):
                predictor.add_new_mask(state, current_frame, id, p)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                masks = masks.squeeze(1)
                segments.add_masks(frame_idx, object_ids, masks)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
                masks = masks.squeeze(1)
                if frame_idx == current_frame:
                    continue
                segments.add_masks(frame_idx, object_ids, masks)
    save_smap(segments, entities)
    return segments, entities

def get_entity_image(images, mask):
    pass

def extract_semantics(images: torch.Tensor, segments: Segments, entities: Entities):
    global save_path, image_path, mask_generator, predictor, state
    for id, entity in enumerate(entities.container):
        smap = segments.smaps[entity['prompt_frame']]
        mask = smap == id
        image = get_entity_image(images[entity['prompt_frame']], mask)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--image_folder', type=str, default='input')
    parser.add_argument('--save_folder', type=str, default='semantic')
    parser.add_argument('--sam_path', type=str, default="facebook/sam2-hiera-large")
    torch.set_default_dtype(torch.float32)
    return parser.parse_args()

def main() -> None:
    global save_path, image_path, mask_generator, predictor, state
    seed_everything(42)
    args = prepare_args()
    image_path = os.path.join(args.dataset_path, args.image_folder)
    save_path = os.path.join(args.dataset_path, args.save_folder)
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(args.sam_path, 
                                                                # points_per_side=32,
                                                                # pred_iou_thresh=0.7,
                                                                # box_nms_thresh=0.7,
                                                                # stability_score_thresh=0.85,
                                                                # crop_n_layers=1,
                                                                # crop_n_points_downscale_factor=1,
                                                                min_mask_region_area=100
                                                                )
    predictor = SAM2VideoPredictor.from_pretrained(args.sam_path)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(image_path)
    img_list = []
    WARNED = False
    images = [os.path.join(image_path, p) for p in os.listdir(image_path)]
    images.sort()
    for image_file in images:
        image = cv2.imread(image_file)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img[None, ...] for img in img_list]
    images = torch.cat(images)

    os.makedirs(save_path, exist_ok=True)
    segments, entities = video_segment(images)
    extract_semantics(images, segments, entities)

if __name__  == '__main__':
    main()