import os
import io
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

def show_mask(mask, ax: matplotlib.axes.Axes, obj_id=None, random_color=False):
    ax.clear()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    img = ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

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
    points = np.array([[2, 326], [300, 300]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_dir)

        # add new prompts and instantly get the output on the same frame
        # frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, ann_frame_idx, ann_obj_id, points, labels)
        for id, mask in enumerate(masks):
            frame_idx, object_ids, masks = predictor.add_new_mask(state, ann_frame_idx, id, mask)

        fig, ax = plt.subplots()

        # ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        # show_points(points, labels, ax)
        # show_mask((masks[0] > 0.0).cpu().numpy(), ax, obj_id=object_ids[0])
        # plt.show()

        # propagate the prompts to get masklets throughout the video
        color = torch.rand((len(object_ids)+1, 3), device='cuda')
        color[-1] = torch.zeros(3)
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            # print(frame_idx, object_ids, masks.shape)
            map = torch.full(masks.shape[-2:], -1, device='cuda')
            for id, mask in zip(object_ids, masks, strict=True):
                mask = mask[0]>0
                # assert torch.all(map[mask]==-1).item()
                map[mask] = id
            image = color[map]
            torchvision.utils.save_image(image.permute(2,0,1), f'data/lerf/waldo_kitchen/temp/{str(frame_idx).rjust(5,'0')}.jpg')
            # show_mask((masks[0] > 0.0).cpu().numpy(), ax, obj_id=object_ids[0], random_color=True)
            # plt.show()

def mask():
    img_path = 'data/lerf/waldo_kitchen/input/00000.jpg'
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large", 
                                                                # points_per_side=32,
                                                                # pred_iou_thresh=0.7,
                                                                # box_nms_thresh=0.7,
                                                                # stability_score_thresh=0.85,
                                                                crop_n_layers=1,
                                                                crop_n_points_downscale_factor=1,
                                                                min_mask_region_area=100
                                                                )
    masks = mask_generator.generate(image)
    smap = np.zeros_like(image)
    for mask in masks:
        mask=mask['segmentation']
        color = np.random.randint((128,128,128), dtype=np.uint8)
        smap[mask]=color
    # plt.imshow(smap)
    # plt.show()
    track([m['segmentation'] for m in masks])

def main() -> None:
    mask()

main()