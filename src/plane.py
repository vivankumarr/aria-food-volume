import numpy as np


def backproject_depth_map(depth_map, fx, fy, cx, cy):
    """ Convert a depth map into an array of 3D coordinates in the rectified left CV camera's coordinate frame. """
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

    # Rearranged pinhole model formulas
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map

    # Shape is (H, W, 3)
    return np.stack([x, y, z], axis=-1)


def depth_to_world_points(depth_map, pose_matrix, intrinsics):
    """ Backproject one depth map into 3D and place it in the shared world frame. """
    points_cam = backproject_depth_map(depth_map, *intrinsics)

    # Apply the rigid transform to convert from rect left frame to world frame
    return points_cam @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]


def fit_plane_ransac(points, inlier_threshold_m=0.002, num_iterations=300, seed=0):
    """
    Find the plane supported by the largest number of selected points. Also return a boolean
    mask array indicating which input points supported the final plane.
    """
    rng = np.random.default_rng(seed)
    best_inlier_mask, best_count = None, -1

    for _ in range(num_iterations):
        # Take 3 random points, fit a plane through them, keep the best plane
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]

        normal = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(normal)

        # If the points are collinear, they don't define a plane so skip
        if length < 1e-9:
            continue
        normal = normal / length
        offset = -normal @ p0

        # Compute signed perpendicular distance of all points to this plane and count inliers
        inlier_mask = np.abs(points @ normal + offset) < inlier_threshold_m
        if inlier_mask.sum() > best_count:
            best_count, best_inlier_mask = inlier_mask.sum(), inlier_mask

    inlier_points = points[best_inlier_mask]
    centroid = inlier_points.mean(axis=0)
    centered = inlier_points - centroid

    # PCA to get direction of smallest point spread (i.e., fit plane's normal)
    scatter_matrix = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(scatter_matrix)
    normal = eigenvectors[:, 0]
    offset = -normal @ centroid

    if offset < 0:
        normal = -normal
        offset = -offset

    return normal, offset, best_inlier_mask


def build_plane_frame(plane_normal, plane_offset):
    """ Build the transform matrix T_plane_world, where resting plane is z = 0 (+z is up). """
    helper_vec = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(helper_vec, plane_normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(plane_normal, x_axis)

    # n . (-offset * n) + offset = -offset * (n . n) + offset = 0 so use -offset * n as simple plane origin
    plane_origin = -plane_offset * plane_normal

    # Build the rigid transform
    T_world_plane = np.eye(4)
    T_world_plane[:3, 0] = x_axis
    T_world_plane[:3, 1] = y_axis
    T_world_plane[:3, 2] = plane_normal
    T_world_plane[:3, 3] = plane_origin

    return np.linalg.inv(T_world_plane)