import pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

def dim2vertexes(dim, center=[0,0,0]):
    width, height, depth = dim
    cx, cy, cz = center 
    # X axis point to the right
    right = cx + width / 2.0
    left = cx - width / 2.0
    # Y axis point downward
    top = cy - height / 2.0
    bottom = cy + height / 2.0
    # Z axis point forward
    front = cz + depth / 2.0
    rear = cz - depth / 2.0
    
    # List of 8 vertices of the box       
    vertices = [
        [right, top, front],    # Front Top Right
        [left, top, front],     # Front Top Left
        [left, bottom, front],  # Front Bottom Left
        [right, bottom, front], # Front Bottom Right
        [right, top, rear],     # Rear Top Right
        [left, top, rear],      # Rear Top Left
        [left, bottom, rear],   # Rear Bottom Left
        [right, bottom, rear],  # Rear Bottom Right
    ]
    
    vertices = np.array(vertices)
    
    return vertices

def draw_cube(img, points, color=(255, 0, 0), thickness=3):
    """
    Draws cube with a thick solid line across
    the front top edge and an X on the top face.
    """
    # draw front
    cv2.line(img, points[0], points[1], color, thickness)
    cv2.line(img, points[1], points[2], color, thickness)
    cv2.line(img, points[3], points[2], color, thickness)
    cv2.line(img, points[3], points[0], color, thickness)

    # draw back
    cv2.line(img, points[4], points[5], color, thickness)
    cv2.line(img, points[6], points[5], color, thickness)
    cv2.line(img, points[6], points[7], color, thickness)
    cv2.line(img, points[4], points[7], color, thickness)

    # draw sides
    cv2.line(img, points[0], points[4], color, thickness)
    cv2.line(img, points[7], points[3], color, thickness)
    cv2.line(img, points[5], points[1], color, thickness)
    cv2.line(img, points[2], points[6], color, thickness)


def draw_pose_vertex(img, obj_dim, r_vec, t_vec, K, distrot=np.array([0.,0.,0.,0.])):
    vertices = dim2vertexes(obj_dim)
    projected_vertices, _ = cv2.projectPoints(vertices, r_vec, t_vec, K, distrot)
    projected_vertices = projected_vertices.reshape((8,2)).astype(np.int32)
    color = tuple(np.random.uniform([0,0,0], [255,255,255]))
    img_draw = img.copy()
    draw_cube(img_draw, projected_vertices, color, thickness=2)
    return img_draw


def draw_box(img, lefttop, rightbottom, thickness=3):
    color = tuple(np.random.uniform([0,0,0], [255,255,255]))
    img_draw = img.copy()
    cv2.rectangle(img_draw, lefttop, rightbottom, color, thickness)
    return img_draw
    
    
def coor2pose(coor, c_box, s_box, out_res, mask, K):
    """_summary_

    Args:
        coor (np.array): 3d coordinates
        c_box (np.array): centor of bounding box
        s_box (int): size of bounding box
        out_res(int): out coor imgs size
    """
    s = int(s_box)
    wh_begin = c_box - s / 2  
    wh_unit = s / out_res  

    wh_arange = np.arange(out_res)
    y, x = np.meshgrid(wh_arange, wh_arange)  # (h, w)
    x2d = np.stack((wh_begin[0,None,None]+y*wh_unit, wh_begin[1,None,None]+x*wh_unit), axis=-1)
    x2d = x2d.reshape(-1,2).astype(np.float32)
    coor = coor.astype(np.float32)
    coor = coor.transpose(1, 2, 0)
    coor = coor.reshape(-1, 3)
    b_mask = mask.reshape(-1) > 0.9
    
    x3d = coor[b_mask]
    x2d = x2d[b_mask]
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    _, R, T= cv2.solvePnP(x3d, x2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    
    return R, T

def coor2pose1(coor, c, s, out_res, mask, K):
    select_pts_2d = []
    select_pts_3d = []
    c_w = int(c[0])
    c_h = int(c[1])
    w_begin = c_w - int(s) / 2.
    h_begin = c_h - int(s) / 2.
    w_unit = out_res * 1.0 / s  
    h_unit = out_res * 1.0 / s
    
    for x in range(out_res):
        for y in range(out_res):
            if (coor[x][y] == np.array([0.,0.,0.])).all():
                continue
            select_pts_2d.append([w_begin + y * w_unit, h_begin + x * h_unit])
            select_pts_3d.append(coor[x][y])

    model_points = np.asarray(select_pts_3d, dtype=np.float32)
    image_points = np.asarray(select_pts_2d, dtype=np.float32)
    dist_coeffs = np.zeros((4, 1)) 
    _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    
    return R_vector, T_vector

