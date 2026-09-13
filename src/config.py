import os

N_KEYFRAMES = 40  # Number of keyframes from the VRS recording to use in TSDF fusion

# Only consider depth valid in this distance range
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 1.20

FOOD_KEEPOUT_PX = 10  # Food mask dilation parameter so that food pixels don't contaminate the plate plane fit
MASK_EROSION_PX = 2  # Food mask erosion parameter so that only pixels well inside the food mask are considered during TSDF

# TSDF parameters (voxel grid and boundary ring detection)
VOXEL_SIZE_M = 0.0025
TRUNCATION_M = 4 * VOXEL_SIZE_M
CONTACT_TOLERANCE_MM = 25.0
PINHOLE_MAX_VERTICES = 20
PINHOLE_MAX_SPAN_MM = 12.0

DEPTH_FROM_STEREO_DIR = os.environ.get("DEPTH_FROM_STEREO_DIR", "third_party/projectaria_gen2_depth_from_stereo")
FOUNDATION_STEREO_DIR = DEPTH_FROM_STEREO_DIR + "/FoundationStereo"
FOUNDATION_STEREO_CKPT = os.environ.get("FOUNDATION_STEREO_CKPT", "checkpoints/model_best_bp2.pth")
SAM2_MODEL = "facebook/sam2.1-hiera-large"