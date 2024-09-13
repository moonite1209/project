import os
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
mask_generator = None
predictor = None
state = None
class Segments:
    container: list
    def __init__(self, image_num, image_height, image_width) -> None:
        self.image_num = image_num
        self.image_height = image_height
        self.image_width = image_width
        self.cursor = 0
        self.container = []
        
    def remove_duplicate(self, masks):
        return masks
    
    def append(self, masks):
        pass

class Entities:
    container: list
    def __init__(self, iamge_num) -> None:
        self.container = []

    def remove_duplicate(self, frame_idx, object_ids, masks, prompt):
        for entity in self.container:
            for i, mask in enumerate(masks):
                if iou(entity[frame_idx], mask)>0.8:
                    prompt.pop(object_ids[i])
        return prompt

    def add_entities(self, prompt):
        object_ids = []
        for p in prompt:
            object_ids.append(len(self.container))
            self.container.append({})
        return object_ids

    def add_entity_masks(self, frame_idx, object_ids, masks: torch.Tensor):
        for id, mask in zip(object_ids, masks, strict=True):
            entity = self.container[id]
            entity[frame_idx] = mask.squeeze(0)


def track(masks: Sequence[torch.Tensor]) -> None:
    video_dir = 'data/lerf/waldo_kitchen/input'
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)
    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[300, 300]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_dir)

        # add new prompts and instantly get the output on the same frame
        # frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, ann_frame_idx, ann_obj_id, points, labels)
        for id, mask in enumerate(masks):
            frame_idx, object_ids, masks = predictor.add_new_mask(state, ann_frame_idx, id, mask)
        # frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 10, 0, points, labels)
        # propagate the prompts to get masklets throughout the video
        color = torch.rand((len(object_ids)+1, 3), device='cuda')
        color[-1] = torch.zeros(3)
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0):
            print(frame_idx, object_ids, masks.shape)
            map = torch.full(masks.shape[-2:], -1, device='cuda')
            for id, mask in zip(object_ids, masks, strict=True):
                mask = mask[0]>0
                # assert torch.all(map[mask]==-1).item()
                map[mask] = id
            image = color[map]
            torchvision.utils.save_image(image.permute(2,0,1), f'data/lerf/waldo_kitchen/temp/{str(frame_idx).rjust(5,'0')}.jpg')

def mask():
    img_path = 'data/lerf/waldo_kitchen/input/00000.jpg'
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large", 
                                                                # points_per_side=32,
                                                                # pred_iou_thresh=0.7,
                                                                # box_nms_thresh=0.7,
                                                                # stability_score_thresh=0.85,
                                                                # crop_n_layers=1,
                                                                # crop_n_points_downscale_factor=1,
                                                                min_mask_region_area=100
                                                                )
    masks = mask_generator.generate(image)
    smap = np.zeros_like(image)
    for mask in masks:
        mask=mask['segmentation']
        color = np.random.randint((256,256,256), dtype=np.uint8)
        smap[mask]=color
    track([m['segmentation'] for m in masks])

def iou(mask1, mask2):
    mask1=mask1>0
    mask2=mask2>0
    return (mask1 & mask2).sum()/(mask1 | mask2).sum()

def get_entities(frame_idx, prompt):
    global image_path, predictor, state
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(image_path)
        # predictor.reset_state(state)
        for id, p in enumerate(prompt):
            predictor.add_new_mask(state, frame_idx, id, p)
        for frame_index, object_ids, masks in predictor.propagate_in_video(state): # masks: (n, 1, h, w)
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

def get_video_masks(frame_num, start_frame_index, masks):
    global image_path, predictor, state
    out_masks = [None for i in range(frame_num)]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(image_path)
        # predictor.reset_state(state)
        for id, mask in enumerate(masks):
            predictor.add_new_mask(state, start_frame_index, id, mask)
        for frame_index, object_ids, masks in predictor.propagate_in_video(state):
            if frame_index == start_frame_index:
                continue
            out_masks[frame_index] = masks
        for frame_index, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
            if frame_index == start_frame_index:
                continue
            out_masks[frame_index] = masks
    return out_masks


def video_segment(images: np.ndarray):
    global image_path, mask_generator, predictor, state
    segments = Segments(len(images), images.shape[1], images.shape[2])
    entities  =Entities(len(images))
    for current_frame, image in tqdm(enumerate(images), desc='video_segment'):
        prompt = get_prompt(image)
        frame_idx, object_ids, masks = get_entities(current_frame, prompt)
        prompt = entities.remove_duplicate(frame_idx, object_ids, masks, prompt)
        ids = entities.add_entities(prompt)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(image_path)
            # predictor.reset_state(state)
            for id, p in zip(ids, prompt, strict=True):
                predictor.add_new_mask(state, current_frame, id, p)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                entities.add_entity_masks(frame_idx, object_ids, masks)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
                if frame_idx == current_frame:
                    continue
                entities.add_entity_masks(frame_idx, object_ids, masks)
    return entities
        
def extract_semantics(images: np.ndarray, save_folder: str):
    global image_path, mask_generator, predictor, state

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
    global image_path, mask_generator, predictor, state
    seed_everything(42)
    args = prepare_args()
    image_path = os.path.join(args.dataset_path, args.image_folder)
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
    for image_path in images:
        image = cv2.imread(image_path)

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

    save_folder = os.path.join(args.dataset_path, args.save_folder)
    os.makedirs(save_folder, exist_ok=True)
    video_segment(images)
    extract_semantics(images, save_folder)

if __name__  == '__main__':
    main()