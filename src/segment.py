import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
import config


FOOD_ID, PLATE_ID = 1, 2

def write_frames(rectified_left_images, frames_dir):
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)

    for keyframe_idx, rect_left in enumerate(rectified_left_images):
        # Convert the monochrome images into 3-channel RGB and save as jpg for SAM 2 video mode
        rgb_frame = np.stack([rect_left] * 3, axis=-1)
        Image.fromarray(rgb_frame).save(f"{frames_dir}/{keyframe_idx:05d}.jpg", quality=95)


def save_prompt_grid(rect_left, out_path):
    """ Keyframe 0 with a pixel grid to read off prompt coordinates (temp substitute for a gaze prompt). """
    plt.figure(figsize=(9, 9))
    plt.imshow(rect_left, cmap="gray")
    plt.xticks(np.arange(0, rect_left.shape[1], 32))
    plt.yticks(np.arange(0, rect_left.shape[0], 32))
    plt.grid(color="red", alpha=0.3, linewidth=0.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def segment_scan(rectified_left_images, food_prompt_xy, plate_prompt_xy, frames_dir):
    write_frames(rectified_left_images, frames_dir)
    n_keyframes, image_height, image_width = rectified_left_images.shape

    predictor = SAM2VideoPredictor.from_pretrained(config.SAM2_MODEL).cuda()
    # Initialize the frame index, allocate SAM 2 memory bank, etc.
    inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True)
    prompt_points = np.array([food_prompt_xy, plate_prompt_xy], dtype=np.float32)

    # Point prompt for food mask, negative click prompt on plate so they don't mix
    predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=FOOD_ID, points=prompt_points,
                                    labels=np.array([1, 0], dtype=np.int32))

    # Point prompt for plate mask
    predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=PLATE_ID, points=prompt_points,
                                    labels=np.array([0, 1], dtype=np.int32))

    food_masks = np.zeros((n_keyframes, image_height, image_width), dtype=bool)
    plate_masks = np.zeros((n_keyframes, image_height, image_width), dtype=bool)

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        for slot, obj_id in enumerate(obj_ids):
            mask = (mask_logits[slot, 0] > 0.0).cpu().numpy()
            if obj_id == FOOD_ID:
                food_masks[frame_idx] = mask
            else:
                plate_masks[frame_idx] = mask

    return food_masks, plate_masks