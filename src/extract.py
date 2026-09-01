import sys
import numpy as np
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps import get_eyegaze_point_at_depth
import config

sys.path.insert(0, config.DEPTH_FROM_STEREO_DIR)
import stereo_utils


def extract_scan(vrs_path, n_keyframes=config.N_KEYFRAMES):
    """ Take a VRS file as input and return the keyframe images (as rectified stereo pairs), poses, and pinhole intrinsics. """
    provider = data_provider.create_vrs_data_provider(vrs_path)
    left_id = provider.get_stream_id_from_label("slam-front-left")
    right_id = provider.get_stream_id_from_label("slam-front-right")
    hf_vio_id = provider.get_stream_id_from_label("vio_high_frequency")

    all_frame_timestamps_ns = np.array(provider.get_timestamps_ns(left_id, TimeDomain.DEVICE_TIME))
    vio_timestamps_ns = np.array(provider.get_timestamps_ns(hf_vio_id, TimeDomain.DEVICE_TIME))

    # Only keep frames that were present after VIO was initialized
    vio_covered = all_frame_timestamps_ns >= vio_timestamps_ns[0]
    frame_source_indices = np.nonzero(vio_covered)[0]
    frame_timestamps_ns = all_frame_timestamps_ns[vio_covered]

    # Get camera positions from VIO pose sampled closest to each frame's timestamp
    camera_positions = np.array([provider.get_vio_high_freq_data_by_time_ns(hf_vio_id, int(timestamp_ns), TimeDomain.DEVICE_TIME, 
                                                                            TimeQueryOptions.CLOSEST).transform_odometry_device.translation().ravel()
                                                                            for timestamp_ns in frame_timestamps_ns
                               ])

    # Get displacement vectors for each change in position between consecutive frames
    step_vectors = np.diff(camera_positions, axis=0)
    step_lengths = np.linalg.norm(step_vectors, axis=1)

    # Convert the values into distance walked so far
    distance_traveled = np.concatenate([[0.0], np.cumsum(step_lengths)])
    total_distance = distance_traveled[-1]

    target_distances = np.linspace(0.0, total_distance, n_keyframes)
    keyframe_indices = np.searchsorted(distance_traveled, target_distances)
    keyframe_timestamps_ns = frame_timestamps_ns[keyframe_indices]

    # keyframe_indices stores the 0-indexed frame indices in the filtered frame list, but we need the indices from the original frame list
    keyframe_frame_indices = frame_source_indices[keyframe_indices]
    print(f"{total_distance:.2f} m walked. Selected {len(keyframe_indices)} keyframes.")

    keyframe_left_images = []
    keyframe_right_images = []

    for frame_idx in keyframe_frame_indices:
        left_img, _ = provider.get_image_data_by_index(left_id, int(frame_idx))
        right_img, _ = provider.get_image_data_by_index(right_id, int(frame_idx))

        # All images are (512, 512) single-channel
        keyframe_left_images.append(left_img.to_numpy_array())
        keyframe_right_images.append(right_img.to_numpy_array())

    # Stack into (21, 512, 512) shape
    keyframe_left_images = np.array(keyframe_left_images)
    keyframe_right_images = np.array(keyframe_right_images)

    # Aria Gen 2 calibration objects
    device_calib = provider.get_device_calibration()
    left_calib = device_calib.get_camera_calib("slam-front-left")
    right_calib = device_calib.get_camera_calib("slam-front-right")

    T_leftCam_device = left_calib.get_transform_device_camera().inverse()
    T_rightCam_device = right_calib.get_transform_device_camera().inverse()

    # Create pinhole camera using stereo_utils helpers. Keep original principal point since it'll be the same
    image_height, image_width = keyframe_left_images.shape[1:3]
    rect_calib = stereo_utils.fisheye_to_linear_calib(left_calib, focal_scale=1.25, output_width=image_width, output_height=image_height, 
                                                      use_original_pp=True)
    fx, fy, cx, cy = rect_calib.get_projection_params()

    # Get both rotations needed to turn the cameras into shared, rectified orientation
    R_left_rect, R_right_rect = stereo_utils.create_scanline_rectified_cameras(T_leftCam_device, T_rightCam_device)

    rectified_left_images = []
    rectified_right_images = []
    for left_img, right_img in zip(keyframe_left_images, keyframe_right_images):
        rect_left, rect_right = stereo_utils.rectify_stereo_pair(left_img, right_img, left_calib, right_calib, rect_calib, rect_calib,
                                                                 R_left_rect, R_right_rect, InterpolationMethod.BILINEAR)
        rectified_left_images.append(np.squeeze(rect_left))
        rectified_right_images.append(np.squeeze(rect_right))

    rectified_left_images = np.array(rectified_left_images)
    rectified_right_images = np.array(rectified_right_images)

    keyframe_poses = []
    for timestamp_ns in keyframe_timestamps_ns:
        vio_sample = provider.get_vio_high_freq_data_by_time_ns(hf_vio_id, int(timestamp_ns), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
        T_world_device = vio_sample.transform_odometry_device
        # Chain poses and get transform from rectified left cam to world for each keyframe
        T_world_rectCam = stereo_utils.compute_T_world_rectCam(T_world_device, T_leftCam_device, R_left_rect)
        keyframe_poses.append(T_world_rectCam.to_matrix())

    # Get gaze prompt for SAM 2 on food item
    gaze_id = provider.get_stream_id_from_label("eyegaze")
    gaze = provider.get_eye_gaze_data_by_time_ns(gaze_id, int(keyframe_timestamps_ns[0]), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
    vio0 = provider.get_vio_high_freq_data_by_time_ns(hf_vio_id, int(keyframe_timestamps_ns[0]), TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)

    # CPF --> rectified left camera
    T_rectCam_cpf = (np.linalg.inv(keyframe_poses[0]) @ vio0.transform_odometry_device.to_matrix() @ device_calib.get_transform_device_cpf().to_matrix())
    
    if gaze.spatial_gaze_point_valid:
        p_cpf = np.asarray(gaze.spatial_gaze_point_in_cpf, dtype=float).ravel()
    else:
        # Vergence points not recorded for some reason in most files, use 25 cm along gaze ray as the placeholder point, works identically
        p_cpf = np.asarray(get_eyegaze_point_at_depth(gaze.yaw, gaze.pitch, 0.25), dtype=float).ravel()
    
    p_cam = T_rectCam_cpf[:3, :3] @ p_cpf + T_rectCam_cpf[:3, 3]
    # Perspective projection from rectified left camera to 2D image
    gaze_xy = (int(round(fx * p_cam[0] / p_cam[2] + cx)), int(round(fy * p_cam[1] / p_cam[2] + cy)))
    
    return {
        "rectified_left_images": rectified_left_images,
        "rectified_right_images": rectified_right_images,
        "keyframe_poses": np.array(keyframe_poses),
        "intrinsics": (fx, fy, cx, cy),
        "image_shape": (image_height, image_width),
        "baseline_m": stereo_utils.compute_stereo_baseline(T_leftCam_device, T_rightCam_device),
        "gaze_xy": gaze_xy,
        "total_distance_m": total_distance
    }