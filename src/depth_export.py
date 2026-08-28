import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
import config

os.environ["XFORMERS_DISABLED"] = "1"
sys.path.insert(0, config.DEPTH_FROM_STEREO_DIR)
sys.path.insert(0, config.FOUNDATION_STEREO_DIR)
import stereo_utils
from core.foundation_stereo import FoundationStereo


def load_foundation_stereo():
    stereo_cfg = OmegaConf.load(os.path.join(os.path.dirname(config.FOUNDATION_STEREO_CKPT), "cfg.yaml"))
    stereo_cfg["vit_size"] = "vitl"  # VIT-L tends to perform better on the Aria monochrome images
    stereo_cfg.valid_iters = 32

    stereo_model = FoundationStereo(stereo_cfg)
    checkpoint = torch.load(config.FOUNDATION_STEREO_CKPT, map_location="cpu", weights_only=False)
    stereo_model.load_state_dict(checkpoint["model"])

    # Switch layers like norm and dropout to inference behavior
    return stereo_model.cuda().eval(), stereo_cfg


def compute_disparity(stereo_model, stereo_cfg, rect_left, rect_right):
    """ Use FoundationStereo to compute a dense disparity map for each pair of rectified images. """
    left_rgb = np.stack([rect_left] * 3, axis=-1)
    right_rgb = np.stack([rect_right] * 3, axis=-1)

    # Change arrays from HWC to BCHW
    left_tensor = torch.from_numpy(left_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)
    right_tensor = torch.from_numpy(right_rgb).float().cuda().permute(2, 0, 1).unsqueeze(0)

    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            disparity = stereo_model.forward(left_tensor, right_tensor, iters=stereo_cfg.valid_iters, test_mode=True)

    return disparity.float().cpu().numpy().squeeze()


def compute_keyframe_depths(rectified_left_images, rectified_right_images, baseline_m, fx):
    stereo_model, stereo_cfg = load_foundation_stereo()

    keyframe_depths = []
    for rect_left, rect_right in zip(rectified_left_images, rectified_right_images):
        disparity = compute_disparity(stereo_model, stereo_cfg, rect_left, rect_right)
        # Depth = (Baseline * Focal Length) / Disparity
        keyframe_depths.append(stereo_utils.disparity_to_depth(disparity, baseline_m, fx))

    del stereo_model
    torch.cuda.empty_cache()
    # Stack all depth maps and return one array
    return np.array(keyframe_depths)