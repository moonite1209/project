from dataclasses import dataclass, field
import os
import gc
import logging
import random
import shutil
import io
import pickle
import argparse
import sys
from typing import List, Sequence, Tuple, Type
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
from tqdm import tqdm
import open_clip

image_path = None
save_path = None
device = torch.device('cuda:0')
tb_writer = None

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            device='cuda',
            precision='fp16'
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

class Segments:
    smaps: List[np.ndarray]
    def __init__(self, image_num, image_height, image_width) -> None:
        self.image_num = image_num
        self.image_height = image_height
        self.image_width = image_width
        self.cursor = 0
        self.smaps = [np.full((image_height, image_width), -1, dtype=np.int32) for i in range(image_num)]
        
    def remove_duplicate(self, frame_idx, object_ids, masks, prompt: list):
        smap = self.smaps[frame_idx]
        ret = []
        for i, mask in enumerate(masks):
            if duplicate(smap, mask)<0.8:
                ret.append(prompt[i])
        print(f'remove {len(prompt)-len(ret)} at {frame_idx}')
        return ret
    
    def add_masks(self, frame_idx, object_ids, masks):
        smap=self.smaps[frame_idx]
        for id, mask in zip(object_ids, masks, strict=True):
            smap[mask] = id

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
        print(f'add {len(object_ids)} at {current_frame} total {len(self.container)} size {sys.getsizeof(self)}')
        return object_ids

    def get_colormap(self):
        colormap=[torch.rand(3) for i in range(len(self.container))]
        colormap.append(torch.zeros(3))
        return torch.stack(colormap).cuda()
    
def duplicate(smap, mask):
    smap=smap>=0
    mask=mask>0
    return (smap&mask).sum()/mask.sum()

def calculate_iou(mask1, mask2):
    # 计算两个 mask 的交集和并集
    mask1 = mask1>0
    mask2 = mask2>0
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    return intersection / union if union > 0 else 0

def save_smap(segments: Segments, entities: Entities):
    colormap = entities.get_colormap()
    for i, smap in enumerate(segments.smaps):
        fmap=colormap[smap]
        torchvision.utils.save_image(fmap.permute(2,0,1), os.path.join(save_path, f'{str(i).rjust(5,'0')}.jpg'))

def get_entities(predictor: SAM2VideoPredictor, state, frame_idx, prompt):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # state = predictor.init_state(image_path)
        predictor.reset_state(state)
        for id, p in enumerate(prompt):
            predictor.add_new_mask(state, frame_idx, id, p)
        for frame_index, object_ids, masks in predictor.propagate_in_video(state): # masks: (n, 1, h, w)
            masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
            if frame_index == frame_idx:
                break
    return frame_index, object_ids, masks

def prompt_filter(mask):
    mask=mask['segmentation']
    x = np.zeros_like(mask)
    x[0, ...] = True
    x[-1, ...] = True
    x[..., 0] = True
    x[..., -1] = True
    return ~np.any(mask & x)

def get_prompt(mask_generator: SAM2AutomaticMaskGenerator, image: np.ndarray):
    records=mask_generator.generate(image) # 2.5G
    records = remove_duplicate_prompt(records)
    return [record['segmentation'] for record in records if prompt_filter(record)]

def combine_records(record1, record2):
    return {
        'segmentation': record1['segmentation']|record2['segmentation']
    }

def remove_duplicate_prompt(records:list):
    # 存储有效的 mask
    unique_records = []
    
    for i, record in enumerate(records):
        # 检查当前 mask 是否与 unique_masks 中的任何 mask 重叠
        is_duplicate = False
        for idx, unique_record in enumerate(unique_records):
            if calculate_iou(record['segmentation'], unique_record['segmentation']) > 0.8:
                is_duplicate = True
                unique_records[idx] = combine_records(record, unique_record)
        
        if not is_duplicate:
            unique_records.append(record)
    
    return unique_records


def prompt_filter_bbox(record):
    bbox=record['bbox']
    x,y,w,h = bbox
    image_width, image_height = record['segmentation'].shape
    if x!=0 and y!=0 and x+w!=image_width and y+h!=image_height:
        return True
    return False

def get_prompt_bbox(image: torch.Tensor):
    global mask_generator
    records=mask_generator.generate(image.cpu().numpy())
    ret= [record['bbox'] for record in records if prompt_filter_bbox(record)]
    return ret

def mask_or(*masks):
    masks = [m>0 for m in masks]
    ret=torch.zeros_like(masks[0])
    for m in masks:
        ret=ret|m
    return ret

def video_segment(image_names: List[str], images: np.ndarray):
    global image_path, save_path, args

    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(args.sam_path, 
                                                                # points_per_side=32,
                                                                # pred_iou_thresh=0.7,
                                                                # box_nms_thresh=0.7,
                                                                # stability_score_thresh=0.85,
                                                                # crop_n_layers=1,
                                                                # crop_n_points_downscale_factor=1,
                                                                min_mask_region_area=100
                                                                ) # 1G
    predictor = SAM2VideoPredictor.from_pretrained(args.sam_path) #1G
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(image_path) # 3G

    segments = Segments(len(images), images.shape[1], images.shape[2])
    entities  =Entities(len(images))
    for current_frame, image_name, image in tqdm(zip(range(len(images)), image_names, images, strict=True), desc='video_segment'):
        prompt = get_prompt(mask_generator, image)
        if len(prompt)==0:
            continue
        frame_idx, object_ids, masks = get_entities(predictor, state, current_frame, prompt)
        prompt = segments.remove_duplicate(frame_idx, object_ids, masks, prompt)
        if len(prompt)==0:
            continue
        frame_idx, object_ids, masks = get_entities(predictor, state, current_frame, prompt)
        ids = entities.add_entities(frame_idx, object_ids, masks, prompt)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # state = predictor.init_state(image_path)
            predictor.reset_state(state)
            for id, p in zip(ids, prompt, strict=True):
                predictor.add_new_mask(state, current_frame, id, p)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                masks = masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
                segments.add_masks(frame_idx, object_ids, masks)
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state, reverse=True):
                masks = masks = (masks.squeeze(1)>0).clone().detach().cpu().numpy()
                if frame_idx == current_frame:
                    continue
                segments.add_masks(frame_idx, object_ids, masks)
    np.save(os.path.join(save_path, 'segments.npy'), np.stack(segments.smaps))
    for image_name, smap in zip(image_names, segments.smaps, strict=True):
        np.save(os.path.join(save_path, f'{os.path.splitext(image_name)[0]}.npy'), smap)
    with open(os.path.join(save_path, 'segments.pk'), 'wb') as sf, open(os.path.join(save_path, 'entities.pk'), 'wb') as ef:
        pickle.dump(segments, sf)
        pickle.dump(entities, ef)
    return segments, entities

def get_bbox(mask: np.ndarray):
     # 查找掩码中的 True 元素的索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 如果没有 True 元素，则返回全零的边界框
    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)
    
    # 获取边界框的上下左右边界
    x_min, x_max = np.where(rows)[0][[0, -1]] # h
    y_min, y_max = np.where(cols)[0][[0, -1]] # w
    
    # 返回边界框
    return (x_min, y_min, x_max + 1 - x_min, y_max + 1 - y_min) # x, y, h, w

def get_entity_image(image: np.ndarray, mask: np.ndarray):
    image = image.copy()
    # crop by bbox
    x,y,h,w = get_bbox(mask)
    image[~mask] = np.zeros(3, dtype=np.uint8) #分割区域外为白色
    image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
    # pad to square
    l = max(h,w)
    paded_img = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
    else:
        paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
    paded_img = cv2.resize(paded_img, (224,224))
    return paded_img

def extract_semantics(images: np.ndarray, segments: Segments, entities: Entities):
    global save_path, image_path, args
    clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    semantics=[]
    for id, entity in tqdm(enumerate(entities.container), desc='extract semantics'):
        smap = segments.smaps[entity['prompt_frame']]
        mask = smap == id
        entity_image = get_entity_image(images[entity['prompt_frame']], entity['mask'])
        with torch.no_grad():
            semantic = clip.encode_image((torch.from_numpy(entity_image).cuda().permute(2, 0, 1)/255).unsqueeze(0))
        semantic /=semantic.norm(dim=-1, keepdim=True)
        semantics.append(semantic.clone().detach().cpu().numpy())
    semantics = np.concatenate(semantics)
    # semantics = clip.encode_image(entity_images.permute(0, 3, 1, 2))
    np.save(os.path.join(save_path, 'raw_semantics.npy'), semantics)


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
    parser.add_argument('--flag', action='store_true')
    torch.set_default_dtype(torch.float32)
    return parser.parse_args()

def main() -> None:
    global save_path, image_path, args, tb_writer
    seed_everything(42)
    torch.set_default_device(device)
    args = prepare_args()
    image_path = os.path.join(args.dataset_path, args.image_folder)
    save_path = os.path.join(args.dataset_path, args.save_folder)
    img_list = []
    WARNED = False
    image_names = os.listdir(image_path)
    image_names.sort()
    for image_name in image_names:
        image = cv2.imread(os.path.join(image_path, image_name))

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
        img_list.append(image)
    images = np.stack(img_list)

    os.makedirs(save_path, exist_ok=True)
    if args.flag:
        with open(os.path.join(save_path, 'segments.pk'), 'rb') as sf, open(os.path.join(save_path, 'entities.pk'), 'rb') as ef:
            segments = pickle.load(sf)
            entities = pickle.load(ef)
    else:
        segments, entities = video_segment(image_names, images)

    extract_semantics(images, segments, entities)

if __name__  == '__main__':
    main()