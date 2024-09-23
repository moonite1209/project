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
mask_generator = None
predictor = None
clip = None
state = None
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
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
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
    smaps: list
    def __init__(self, image_num, image_height, image_width) -> None:
        self.image_num = image_num
        self.image_height = image_height
        self.image_width = image_width
        self.cursor = 0
        self.smaps = [torch.full((image_height, image_width), -1, device='cuda', dtype=torch.int32) for i in range(image_num)]
        
    def remove_duplicate(self, frame_idx, object_ids, masks, prompt: list):
        smap = self.smaps[frame_idx]
        for i, mask in enumerate(masks):
            if duplicate(smap, mask)>0.8:
                prompt[i]=None
        return [p for p in prompt if p!=None]
    
    def add_masks(self, frame_idx, object_ids, masks):
        smap=self.smaps[frame_idx]
        for id, mask in zip(object_ids, masks, strict=True):
            smap[mask>0][smap[mask>0] == -1] = id

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
    masks=mask_generator.generate(image.cpu().numpy())
    return [torch.from_numpy(mask['segmentation']) for mask in masks if prompt_filter(mask)]


def video_segment(images: torch.Tensor):
    global image_path, mask_generator, predictor, state
    segments = Segments(len(images), images.shape[1], images.shape[2])
    entities  =Entities(len(images))
    for current_frame, image in tqdm(enumerate(images), desc='video_segment'):
        prompt = get_prompt(image)
        if len(prompt)==0:
            continue
        frame_idx, object_ids, masks = get_entities(current_frame, prompt)
        prompt = segments.remove_duplicate(frame_idx, object_ids, masks, prompt)
        if len(prompt)==0:
            continue
        frame_idx, object_ids, masks = get_entities(current_frame, prompt)
        for id, mask in zip(object_ids, masks, strict=True):
            torchvision.utils.save_image((images[current_frame]*(mask>0).unsqueeze(-1)).permute(2,0,1)/255, os.path.join(save_path, 'temp', f'{current_frame}_{id}.jpg'))
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
    torch.save(torch.stack(segments.smaps), os.path.join(save_path, 'segments.pt'))
    with open(os.path.join(save_path, 'segments.pk'), 'wb') as sf, open(os.path.join(save_path, 'entities.pk'), 'wb') as ef:
        pickle.dump(segments, sf)
        pickle.dump(entities, ef)
    return segments, entities

def get_bbox(mask: torch.Tensor):
    coord = mask.argwhere()
    value, indices = coord.min(dim=0)
    minx, miny = value
    value, indices = coord.max(dim=0)
    maxx, maxy = value
    return minx, miny, maxx-minx+1, maxy-miny+1

def get_entity_image(image: torch.Tensor, mask: torch.Tensor):
    image = image.clone().cuda()
    # crop by bbox
    x,y,h,w = get_bbox(mask)
    image[~mask] = torch.zeros(3, dtype=torch.uint8) #分割区域外为白色
    image = image[x:x+h, y:y+w, ...] #将img按分割区域bbox裁剪
    # pad to square
    l = max(h,w)
    paded_img = torch.zeros((l, l, 3), device=device)
    if h > w:
        paded_img[:,(h-w)//2:(h-w)//2 + w, :] = image
    else:
        paded_img[(w-h)//2:(w-h)//2 + h, :, :] = image
    paded_img = torch.from_numpy(cv2.resize(paded_img.cpu().numpy(), (224,224))).cuda()
    return paded_img

def extract_semantics(images: torch.Tensor, segments: Segments, entities: Entities):
    global save_path, image_path, mask_generator, predictor, clip, state
    semantics=[]
    for id, entity in tqdm(enumerate(entities.container), desc='extract semantics'):
        smap = segments.smaps[entity['prompt_frame']]
        mask = smap == id
        entity_image = get_entity_image(images[entity['prompt_frame']], entity['mask']>0)
        semantic = clip.encode_image((entity_image.permute(2, 0, 1)).unsqueeze(0))
        semantics.append(semantic.cpu())
    semantics = torch.stack(semantics)
    # semantics = clip.encode_image(entity_images.permute(0, 3, 1, 2))
    torch.save(semantics, os.path.join(save_path, 'semantics.pt'))


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
    global save_path, image_path, mask_generator, predictor, clip, state, tb_writer
    seed_everything(42)
    torch.set_default_device(device)
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
    tb_writer = SummaryWriter(save_path)
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
    images = torch.cat(images).cuda()

    os.makedirs(save_path, exist_ok=True)
    segments, entities = video_segment(images)
    del mask_generator, predictor, state
    clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    extract_semantics(images, segments, entities)

if __name__  == '__main__':
    main()