""" Collection of camera projection helper functions."""
from typing import Optional
from blenderproc.python.postprocessing.PostProcessingUtility import dist2depth
from blenderproc.python.types.MeshObjectUtility import MeshObject, create_primitive, get_all_mesh_objects

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from blenderproc.python.utility.Utility import KeyFrame
from blenderproc.python.camera.CameraUtility import get_camera_pose, get_intrinsics_as_K_matrix


def depth_via_raytracing(bvh_tree: BVHTree, frame: Optional[int] = None, return_dist: bool = False) -> np.ndarray:
    """ Computes a depth images using raytracing.

    All pixel that correspond to rays which do not hit any object are set to inf.

    :param bvh_tree: The BVH tree to use for raytracing.
    :param frame: The frame number whose assigned camera pose should be used. If None is given, the current frame
                  is used.
    :param return_dist: If True, a distance image instead of a depth image is returned.
    :return: The depth image with shape [H, W].
    """
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y

    # Generate 2D coordinates of all pixels
    y = np.arange(resolution_y)   
    x = np.arange(resolution_x)
    points = np.stack(np.meshgrid(x, y), -1).astype(np.float64)

    # Calc depth at points
    depth = depth_at_points_via_raytracing(bvh_tree, points.reshape(-1, 2), frame, return_dist)
    
    # Reshape back into depth image
    depth = np.reshape(depth, [resolution_y, resolution_x])
    return depth


def depth_at_points_via_raytracing(bvh_tree: BVHTree, points_2d: np.ndarray, frame: Optional[int] = None, return_dist: bool = False) -> np.ndarray:
    """ Computes the depth values at the given 2D points.

    All points that correspond to rays which do not hit any object are set to inf.

    :param bvh_tree: The BVH tree to use for raytracing.
    :param points_2d: An array of N 2D points with shape [N, 2].
    :param frame: The frame number whose assigned camera pose should be used. If None is given, the current frame
                  is used.
    :param return_dist: If True, distance values instead of depth are returned.
    :return: The depth values with shape [N].
    """
    with KeyFrame(frame):
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        cam2world_matrix = cam_ob.matrix_world
        resolution_x = bpy.context.scene.render.resolution_x
        resolution_y = bpy.context.scene.render.resolution_y

        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [(np.array(cam2world_matrix) @ np.append(np.array(v, np.float64), [1]))[:3] for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[3] - frame[0]
        vec_y = frame[1] - frame[0]

        dists = []
        # Go in discrete grid-like steps over plane
        position = cam2world_matrix.to_translation()
        """objs = get_all_mesh_objects()
        obj = [obj for obj in objs if obj.get_name() == "no_building"][0]
        mesh = obj.mesh_as_trimesh()
        ends = []"""
        for p in points_2d:
            # Compute current point on plane
            end = frame[0] + vec_x * (resolution_x - (p[0] + 0.5)) / float(resolution_x) \
                    + vec_y * (p[1] + 0.5) / float(resolution_y)
            # Send ray from the camera position through the current point on the plane
            _, _, _, dist = bvh_tree.ray_cast(position, end - position)
            if dist is None:
                dist = np.inf

            dists.append(dist)
        
        dists = np.array(dists)
        """loc2, indx, _ = mesh.ray.intersects_location(np.array(position)[None].repeat(len(ends), 0), (ends - np.array(position)[None]), multiple_hits=False)
        dists2 = np.ones((len(loc2),), np.float64) * np.inf
        dists2[indx] = np.linalg.norm(loc2 - np.array(position)[None], axis=-1)
        
        print(dists[:10], dists2[:10])
        print("dists", dists.dtype)
        dists = dists2"""

        if not return_dist:
            return dist2depth(dists, points_2d)
        else:
            return dists
    
def unproject_points(points_2d: np.ndarray, depth: np.ndarray, frame: Optional[int] = None, depth_cut_off: float = 1e6) -> np.ndarray:
    """ Unproject 2D points into 3D

    :param points_2d: An array of N 2D points with shape [N, 2].
    :param depth: A list of depth values corresponding to each 2D point, shape [N].
    :param frame: The frame number whose assigned camera pose should be used. If None is given, the current frame
                  is used.
    :param depth_cut_off: All points that correspond to depth values bigger than this threshold will be set to NaN.
    :return: The unprojected 3D points with shape [N, 3].
    """
    # Get extrinsics and intrinsics
    cam2world = get_camera_pose(frame)
    K = get_intrinsics_as_K_matrix()
    K_inv = np.linalg.inv(K)

    # Flip y axis
    points_2d[..., 1] = (bpy.context.scene.render.resolution_y - 1) - points_2d[..., 1]

    # Unproject 2D into 3D
    points = np.concatenate((points_2d, np.ones_like(points_2d[:, :1])), -1)
    with np.errstate(invalid='ignore'):
        points *= depth[:, None]
        points_cam = (K_inv @ points.T).T

    # Transform into world frame
    points_cam[...,2] *= -1
    points_cam = np.concatenate((points_cam, np.ones_like(points[:, :1])), -1)
    points_world = (cam2world @ points_cam.T).T

    points_world[depth > depth_cut_off, :] = np.nan

    return points_world[:, :3]


def project_points(points: np.ndarray, frame: Optional[int] = None) -> np.ndarray:
    """ Project 3D points into the 2D camera image.

    :param points: A list of 3D points with shape [N, 3].
    :param frame: The frame number whose assigned camera pose should be used. If None is given, the current frame
                  is used.
    :return: The projected 2D points with shape [N, 2].
    """
    # Get extrinsics and intrinsics
    cam2world = get_camera_pose(frame)
    K = get_intrinsics_as_K_matrix()
    world2cam = np.linalg.inv(cam2world)

    # Transform points into camera frame
    points = np.concatenate((points, np.ones_like(points[:, :1])), -1)
    points_cam = (world2cam @ points.T).T
    points_cam[...,2] *= -1
    
    # Project 3D points into 2D
    points_2d = (K @ points_cam[:, :3].T).T
    points_2d /= points_2d[:, 2:]
    points_2d = points_2d[:, :2]

    # Flip y axis
    points_2d[..., 1] = (bpy.context.scene.render.resolution_y - 1) - points_2d[..., 1]


    """cam_ob = bpy.context.scene.camera
    cam = cam_ob.data

    cam2world_matrix = cam_ob.matrix_world
    print(cam2world.tolist(), np.array(cam2world_matrix).tolist(), cam2world - cam2world_matrix)
    # Get position of the corners of the near plane
    frame = cam.view_frame(scene=bpy.context.scene)
    print(frame[0])
    print(frame[1])
    print(frame[3])
    # Bring to world space
    frame = [(np.array(cam2world_matrix) @ np.append(np.array(v, np.float64), [1]))[:3] for v in frame]
    vec_x = frame[3] - frame[0]
    vec_y = frame[1] - frame[0]
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y

    print("vec", np.linalg.norm(vec_x), np.linalg.norm(vec_y))

    indx = 500 * 500

    points_2d_isect = isect_line_plane_v3(cam2world[:3, 3], points[indx], np.array(frame[0]), np.cross(vec_x, vec_y))
    print(points_2d_isect, np.dot(points_2d_isect - frame[0], vec_x) / np.dot(vec_x, vec_x), np.dot(points_2d_isect - frame[0], vec_y) / np.dot(vec_y, vec_y))

    points_2d_isect_proj = np.array([-(np.dot(points_2d_isect - frame[0], vec_x) / np.dot(vec_x, vec_x) * resolution_x + 0.5 - resolution_x), np.dot(points_2d_isect - frame[0], vec_y) / np.dot(vec_y, vec_y) * resolution_y - 0.5])
    print(points_2d[indx].tolist(), points_2d_isect_proj.tolist())

    end2 = frame[0] + vec_x * ((resolution_x - (points_2d_isect_proj[0] + 0.5)) / float(resolution_x)) \
                        + vec_y * ((points_2d_isect_proj[1] + 0.5) / float(resolution_y))
    print(end2.tolist(), points_2d_isect)

    end = frame[0] + vec_x * ((resolution_x - (points_2d[indx][0] + 0.5)) / float(resolution_x)) \
                        + vec_y * ((points_2d[indx][1] + 0.5) / float(resolution_y))
    print(end.tolist(), points_2d_isect)"""
    #sdfsdf


    return points_2d

def pointcloud_from_depth(depth: np.ndarray, frame: Optional[int] = None, depth_cut_off: float = 1e6) -> np.ndarray:
    """ Compute a point cloud from a given depth image.

    :param depth: The depth image with shape [H, W].
    :param frame: The frame number whose assigned camera pose should be used. If None is given, the current frame
                  is used.
    :param depth_cut_off: All points that correspond to depth values bigger than this threshold will be set to NaN.
    :return: The point cloud with shape [H, W, 3]
    """    
    # Generate 2D coordinates of all pixels in the given image.
    y = np.arange(depth.shape[0])   
    x = np.arange(depth.shape[1])
    points = np.stack(np.meshgrid(x, y), -1).astype(np.float64)
    # Unproject the 2D points
    return unproject_points(points.reshape(-1, 2), depth.flatten(), frame, depth_cut_off).reshape(depth.shape[0], depth.shape[1], 3)




# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)

    # The segment is parallel to plane.
    return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
    )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
    )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
    )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
    )
