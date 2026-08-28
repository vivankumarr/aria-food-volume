import copy
import numpy as np
import open3d as o3d
import trimesh
from scipy.ndimage import binary_dilation, binary_erosion
import config
from plane import depth_to_world_points, fit_plane_ransac, build_plane_frame


def find_boundary_loops(faces):
    """ 
    Find the ordered vertex rings around every hole (boundary).
    Boundary edges are those that are only used by one triangle.
    """
    # Every single edge in the mesh (w/ duplicates)
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    
    neighbors = {}
    # Build adjacency list of boundary vertices
    for a, b in boundary_edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    unused = {tuple(edge) for edge in boundary_edges}
    loops = []
    while unused:
        # Get an arbitrary unwalked edge from the set
        start_a, start_b = unused.pop()
        loop = [start_a, start_b]
        while True:
            current, previous = loop[-1], loop[-2]
            next_vertex = None
            for candidate in neighbors.get(current, []):
                key = (min(current, candidate), max(current, candidate))
                if candidate != previous and key in unused:
                    next_vertex = candidate
                    # Mark the edge as walked
                    unused.discard(key)
                    break
            
            if next_vertex is None or next_vertex == loop[0]:
                break
                    
            loop.append(next_vertex)
        loops.append(loop)

    # Two vertices can't bound a hole
    return [loop for loop in loops if len(loop) >= 3]


def estimate_volume(keyframe_depths, food_masks, plate_masks, keyframe_poses, intrinsics, image_shape):
    """ All steps after depth computation to compute volume: plate plane fit, TSDF fusion, clip, cap, and integrate using Trimesh. """
    image_height, image_width = image_shape
    fx, fy, cx, cy = intrinsics

    stride_grid = np.zeros((image_height, image_width), dtype=bool)
    stride_grid[::4, ::4] = True  # Stride of 4 since all plate points not necessary
    plate_points_world = []

    for depth_map, food_mask, plate_mask, pose_matrix in zip(keyframe_depths, food_masks, plate_masks, keyframe_poses):
        points_world = depth_to_world_points(depth_map, pose_matrix, intrinsics) 
        depth_ok = (depth_map > config.DEPTH_MIN_M) & (depth_map < config.DEPTH_MAX_M)

        # Dilate the mask by about FOOD_KEEPOUT_PX pixels so the food mask doesn't mess up the plane fit
        food_keepout = binary_dilation(food_mask, np.ones((3, 3), bool), iterations=config.FOOD_KEEPOUT_PX)
        plate_usable = plate_mask & ~food_keepout & stride_grid & depth_ok

        # Get list of world points for plate
        plate_points_world.append(points_world[plate_usable])
    
    # Get list of world points for plate
    plate_points_world = np.vstack(plate_points_world)
    if len(plate_points_world) < 500:
        return {"status": "not enough points for a plane fit"}

    plane_normal, plane_offset, _ = fit_plane_ransac(plate_points_world, inlier_threshold_m=0.003, num_iterations=300)

    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=config.VOXEL_SIZE_M, sdf_trunc=config.TRUNCATION_M, 
                                                               color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)
    dummy_color = o3d.geometry.Image(np.zeros((image_height, image_width, 3), dtype=np.uint8))  # Don't need color for volume estimation

    for depth_map, food_mask, pose_matrix in zip(keyframe_depths, food_masks, keyframe_poses):
        # Opposite of dilation: shrink the food mask so it doesn't get messed up by the plate depth values
        eroded_mask = binary_erosion(food_mask, np.ones((3, 3), bool), iterations=config.MASK_EROSION_PX)
        valid = eroded_mask & (depth_map > config.DEPTH_MIN_M) & (depth_map < config.DEPTH_MAX_M)

        # Create RGBD image for TSDF fusion
        masked_depth = np.where(valid, depth_map, 0.0).astype(np.float32)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(dummy_color, o3d.geometry.Image(np.ascontiguousarray(masked_depth)), 
                                                                        depth_scale=1.0, depth_trunc=config.DEPTH_MAX_M, convert_rgb_to_intensity=False)

        # Pass world-to-camera (inverse of camera-to-world) to o3d
        tsdf_volume.integrate(rgbd_image, o3d_intrinsics, np.linalg.inv(pose_matrix))

    # Marching cubes (walk voxels, find surface crossings, lerp and create triangles)
    fused_mesh = tsdf_volume.extract_triangle_mesh()
    fused_mesh.remove_degenerate_triangles()
    fused_mesh.remove_duplicated_vertices()
    if len(np.asarray(fused_mesh.triangles)) == 0:
        return {"status": "TSDF did not produce a surface"}

    # Keep only the largest connected component (food cover mesh) and remove random blobs in space
    triangle_clusters, cluster_triangle_counts, _ = fused_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_triangle_counts = np.asarray(cluster_triangle_counts)

    largest_cluster = int(np.argmax(cluster_triangle_counts))
    fused_mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster)
    fused_mesh.remove_unreferenced_vertices()

    T_plane_world = build_plane_frame(plane_normal, plane_offset)
    mesh_in_plane = copy.deepcopy(fused_mesh)
    mesh_in_plane.transform(T_plane_world)

    # Cut the triangles that cross z = 0 (table plane)
    clipped = o3d.t.geometry.TriangleMesh.from_legacy(mesh_in_plane).clip_plane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0]).to_legacy()

    # Don't want to modify the vertex/face containers stored in Open3D internal buffer
    vertices = np.asarray(clipped.vertices).copy()
    faces = np.asarray(clipped.triangles).copy()
    if len(faces) == 0:
        return {"status": "nothing left above the resting plane"}

    # Contact rings are unobserved footprint, pinholes are noise from marching cubes, and there may be unobserved regions
    contact_loops, pinhole_loops, unobserved_loops = [], [], []
    for loop in find_boundary_loops(faces):
        loop_mm = 1000 * vertices[loop]
        if np.median(loop_mm[:, 2]) <= config.CONTACT_TOLERANCE_MM:
            contact_loops.append(loop)

        elif len(loop) <= config.PINHOLE_MAX_VERTICES and np.ptp(loop_mm, axis=0).max() <= config.PINHOLE_MAX_SPAN_MM:
            pinhole_loops.append(loop)

        else:
            unobserved_loops.append(loop)

    counts = {"contact_loops": len(contact_loops),
              "pinhole_loops": len(pinhole_loops),
              "unobserved_loops": len(unobserved_loops)}
    
    if unobserved_loops:
        return {"status": f"{len(unobserved_loops)} unobserved region(s)", **counts}

    
    # Drop the contact rings flat onto the plate, then fan-triangulate every ring shut
    for loop in contact_loops:
        vertices[loop, 2] = 0.0

    for loop in contact_loops + pinhole_loops:
        center = vertices[loop].mean(axis=0)
        if loop in contact_loops:
            center[2] = 0.0

        center_index = len(vertices)
        vertices = np.vstack([vertices, center])
        # For each position k around the ring, build one triangle from the center, the vertex at k, and the vertex at k+1
        faces = np.vstack([faces, [[center_index, loop[k], loop[(k + 1) % len(loop)]] for k in range(len(loop))]])

    capped_mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not capped_mesh.is_watertight:
        capped_mesh.merge_vertices()
    if not capped_mesh.is_watertight:
        return {"status": "mesh did not close", **counts}

    # Make sure the triangle windings are ok and all outward so that volume integral works properly
    capped_mesh.fix_normals()
    
    return {
        "status": "ok",
        "volume_ml": float(abs(capped_mesh.volume) * 1e6),
        "mesh": capped_mesh
    }