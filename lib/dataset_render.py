import trimesh
import os
import pyrender
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
import pickle
from mathutils import Vector
from glob import glob
from scipy.spatial.transform import Rotation as R

def calc_emb_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )

    return ProjEmb

def xyz_from_depth(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )

    return ProjEmb

def mask2bbox_xyxy(mask):
    """NOTE: the bottom right point is included"""
    ys, xs = np.nonzero(mask)[:2]
    top_left = [xs.min(), ys.min()]
    bottom_right = [xs.max(), ys.max()]
    return [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]

def crop_xyz(xyz):
    # get the top left point and right bottom point
    x1, y1, x2, y2 = mask2bbox_xyxy(xyz)
    xyxy = [x1, y1, x2, y2]
    return xyz[y1:y2+2, x1:x2+2], xyxy

def save_pickle(data,file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pose(pose_file):
   pose =  np.loadtxt(pose_file)
   R, T = pose[:3,:3], pose[:3, 3]

   return R, T

class CorRender:
    """render object 3d coordinates.
    """
    def __init__(self, K, objs_path, img_w, img_h, obj_type="ply") -> None:
        self.trimesh_dict = {}
        for obj in glob(os.path.join(objs_path, "*", "*"+obj_type)):
            fuze_trimesh  = trimesh.load(obj)
            self.trimesh_dict[obj.split("/")[-2]] = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.scene = pyrender.Scene()
        self.img_w, self.img_h = img_w, img_h
        
        fx, fy, cx , cy = K[0][0], K[1][1],K[0][2], K[1][2]
        self.cam_matrix = K
        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, 0.1, 100)
    
    def get_camera_pose(self, rot, tran):
        rot_c = inv(rot)
        tran_c = rot_c @ -tran 

        # convert world coordinate to opengl coordiante
        t_mat = np.eye(3)
        t_mat[1,1] = -1
        t_mat[2,2] = -1
        rot_ct = rot_c @ t_mat

        camera_pose = np.eye(4)
        camera_pose[:3,:3] = rot_ct
        camera_pose[:3,3] = tran_c
        
        return camera_pose
     
    def render_depth(self, obj, rot, tran):
        self.scene.add(self.trimesh_dict[obj])
        camera_pose = self.get_camera_pose(rot, tran)
        self.scene.add(self.camera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(self.img_w, self.img_h)
        _, depth = r.render(self.scene)
        
        self.scene.clear()
        r.delete()

        return depth

    def render_xyz(self, obj, rot, tran, crop=False):
        depth = self.render_depth(obj, rot, tran)
        xyz = xyz_from_depth(depth, rot, tran, self.cam_matrix)

        if crop:
            xyz, _ = crop_xyz(xyz)
            np.save("test.npy", xyz)

        return xyz.astype(np.float32)

def load_pose(pose_file):
   pose =  np.loadtxt(pose_file)
   R, T = pose[:3,:3], pose[:3, 3]

   return R, T
    
def load_pickle(file):
    with open(file, 'rb') as f:
        src = pickle.load(f) 
        return src

