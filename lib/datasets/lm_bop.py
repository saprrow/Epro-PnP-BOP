"""
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: LineMOD.py
@time: 18-10-24 下午10:24
@desc: load LineMOD dataset
"""

import torch.utils.data as data
import numpy as np
import ref_bop as ref
import cv2
from utils.img import zoom_in, get_edges
from utils.transform3d import prj_vtx_cam
from utils.io import read_pickle, read_json, save_npy
from utils.visual import draw_pose_vertex, draw_box, coor2pose, coor2pose1
import os, sys
from tqdm import tqdm
import utils.fancy_logger as logger
import pickle
from glob import glob 
import random 
from utils.eval import calc_rt_dist_m 
import matplotlib.pyplot as plt
from dataset_render import CorRender
from multiprocessing import Process, Manager




class LM_BOP(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.model_info = self.load_lm_model_info()

        self.cam_K = ref.K
        logger.info('==> initializing {} {} data.'.format(cfg.dataset.name, split))
        self.anno = []
        self.xyz_render = CorRender(ref.K, ref.objs_path, ref.im_w, ref.im_h)  

        # load dataset
        if split == 'test':
            self.cache_test()
        elif split == 'train':
            self.cache_train()
        else:
            raise ValueError
        
        self.num = len(self.anno)

    
    def cache_train(self):
        cache_file = os.path.join(self.cfg.dataset.path, "train_cache.npy")
        if os.path.exists(cache_file):
            logger.info("already cached.")
            self.anno = np.load(cache_file, allow_pickle=True)
            return
        logger.info("==> caching")
        scenes = glob(os.path.join(self.cfg.dataset.path, "train_pbr", "[0-9]*"))
        for scene in scenes:
            self.cache_scene(scene)
        np.save(cache_file, self.anno)
        logger.info("==> Done")
    
    def cache_scene(self, scene):
        poses_json = os.path.join(scene, "scene_gt.json")  
        bboxes_json = os.path.join(scene, "scene_gt_info.json")  

        poses_dict = read_json(poses_json)
        bboxes_dict = read_json(bboxes_json)
        
        workers = 20
        chunk = len(poses_dict) // workers
        counter = 0
        pose_list = []
        pose_cache = {}
        box_list = []
        box_cache = {}
        for idx in poses_dict:
            pose_cache[idx] = poses_dict[idx]
            box_cache[idx] = bboxes_dict[idx]
            counter += 1
            if counter  == chunk:
                box_list.append(box_cache)
                pose_list.append(pose_cache)
                counter = 0
                pose_cache = {}
                box_cache = {}
        box_list.append(box_cache)
        pose_list.append(pose_cache)

        # accelerate caching with multiprocess
        workers = 20
        pool = []
        with Manager() as manager:
            work_list = [manager.list([]) for i in range(workers)]
            for i in range(workers):
                p = Process(target=self.cache_poses, args=(pose_list[i], box_list[i], scene, work_list[i]))
                p.start()
                pool.append(p)
            for p in pool:
                p.join()

            for data in work_list:
                self.anno += list(data)
        
    def cache_poses(self, *args): 
        poses_dict = args[0]
        bboxes_dict = args[1]
        scene = args[2]
        cache_list = args[3]
        pbar = tqdm(poses_dict)
        pbar.set_description("processing scene-{}".format(scene[-6:]))

        for scene_id in pbar:
            poses = poses_dict[scene_id] 
            bboxes = bboxes_dict[scene_id]
        
            scene_name = "{:06d}".format(int(scene_id))
            rgb_path = os.path.join(scene, "rgb", scene_name + ".png")

            cor_dir = os.path.join(scene, "coordinates")
            if not os.path.exists(cor_dir):
                os.mkdir(cor_dir)

            for anno in poses:
                obj_id = int(anno["obj_id"])
                idx = obj_id - 1 # index model name and box annotation
                bbox = bboxes[idx]["bbox_obj"]
                bbox_vis = bboxes[idx]["bbox_visib"]
                if bbox == [-1,-1,-1,-1] or bbox_vis[2]*bbox_vis[3] < 100:
                    continue
                R = anno["cam_R_m2c"]
                T = anno["cam_t_m2c"]
                rot = np.array(R).reshape(3,3)
                tran = np.array(T)
                obj = self.cfg.dataset.classes[idx]

                
                obj_part_name = scene_name + "_{:06d}.png".format(obj_id)
                mask_path = os.path.join(scene, "mask", obj_part_name)
                mask_vis_path = os.path.join(scene, "mask_visib", obj_part_name)
                
                xyz = self.xyz_render.render_xyz(obj, rot, tran)
                xyz_file = scene_name + "_{:06d}.npy".format(obj_id)
                xyz_path = os.path.join(scene, "coordinates", xyz_file)
                
                save_npy(xyz_path, xyz)
                
                xyz_img = xyz*255
                xyz_img = xyz_img.astype(np.uint8)
                cv2.imwrite(xyz_path.replace("npy", "png"), xyz_img)
                
                anno_item = {}
                anno_item["rgb_pth"] = rgb_path
                anno_item["pose"] = np.concatenate((rot, tran[np.newaxis,:].T), axis=1)
                anno_item["box"] = bbox            
                anno_item["box_vis"] = bbox_vis          
                anno_item["mask_pth"] = mask_path          
                anno_item["mask_pth_vis"] = mask_vis_path          
                anno_item['coor_pth'] = xyz_path
                anno_item['data_type'] = "real"
                anno_item['obj'] = obj
                anno_item['obj_id'] = idx
                
                cache_list.append(anno_item)
    

    def load_lm_model_info(self):
        infos = {}
        for i, model_info in enumerate(self.cfg.dataset.model_info):
            infos[i] = {}
            infos[i]['min_x'] = model_info[0]
            infos[i]['min_y'] = model_info[1]
            infos[i]['min_z'] = model_info[2]
            infos[i]['diameter'] = ((model_info[0]/2)**2 + (model_info[1]/2)**2 + model_info[2]**2)**0.5
        return infos


    def xywh_to_cs_dzi(self, xywh, s_ratio, s_max=None, tp='uniform'):
        x, y, w, h = xywh
        if tp == 'gaussian':
            sigma = 1
            shift = truncnorm.rvs(-self.cfg.augment.shift_ratio / sigma, self.cfg.augment.shift_ratio / sigma, scale=sigma, size=2)
            scale = 1+truncnorm.rvs(-self.cfg.augment.scale_ratio / sigma, self.cfg.augment.scale_ratio / sigma, scale=sigma, size=1)
        elif tp == 'uniform':
            scale = 1+self.cfg.augment.scale_ratio * (2*np.random.random_sample()-1)
            shift = self.cfg.augment.shift_ratio * (2*np.random.random_sample(2)-1)
        else:
            raise
        c = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])]) # [c_w, c_h]
        s = max(w, h)*s_ratio*scale
        if s_max != None:
            s = min(s, s_max)
        return c, s

    @staticmethod
    def xywh_to_cs(xywh, s_ratio, s_max=None):
        x, y, w, h = xywh
        c = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        s = max(w, h)*s_ratio
        if s_max != None:
            s = min(s, s_max)
        return c, s

    def denoise_coor(self, coor):
        """
        denoise coordinates by median blur
        """
        coor_blur = cv2.medianBlur(coor, 3)
        edges = get_edges(coor)
        coor[edges != 0] = coor_blur[edges != 0]
        return coor

    def norm_coor(self, coor, obj_id):
        """
        normalize coordinates by object size
        """
        coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
        coor_x = coor_x / abs(self.model_info[obj_id]['min_x'])
        coor_y = coor_y / abs(self.model_info[obj_id]['min_y'])
        coor_z = coor_z / abs(self.model_info[obj_id]['min_z'])
        return np.dstack((coor_x, coor_y, coor_z))

    def unnorm_coor(self, coor, obj_id):
        """
        normalize coordinates by object size
        """
        coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
        coor_x = coor_x * abs(self.model_info[obj_id]['min_x'])
        coor_y = coor_y * abs(self.model_info[obj_id]['min_y'])
        coor_z = coor_z * abs(self.model_info[obj_id]['min_z'])
        return np.dstack((coor_x, coor_y, coor_z))

    def c_rel_delta(self, c_obj, c_box, wh_box):
        """
        compute relative bias between object center and bounding box center
        """
        c_delta = np.asarray(c_obj) - np.asarray(c_box)
        c_delta /= np.asarray(wh_box)
        return c_delta

    def d_scaled(self, depth, s_box, res):
        """
        compute scaled depth
        """
        r = float(res) / s_box
        return depth / r

    def __getitem__(self, idx):
        if self.split == 'train':
            anno = self.anno[idx]
            obj = anno["obj"]
            obj_id = anno["obj_id"]
            box = anno["box_vis"]
            pose = anno["pose"]
            rgb = cv2.imread(anno["rgb_pth"])
            mask = cv2.imread(anno["mask_pth_vis"], cv2.IMREAD_GRAYSCALE)
            coor = np.load(anno["coor_pth"])

            # xywh = (box[0]-box[2], box[1]-box[3], box[2], box[3])
            xywh = box
            if (self.split == 'train') and self.cfg.dataiter.dzi:
                c, s = self.xywh_to_cs_dzi(xywh, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            else:
                c, s = self.xywh_to_cs(xywh, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            
            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            mask = mask.astype(np.float32) / 255  #change mask label from 255 to 1
            coor = coor * mask[:,:,np.newaxis]
            mask, *_ = zoom_in(mask, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)
            c = np.array([c_w_, c_h_])
            s = s_
            coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            inp = rgb
            out = np.concatenate([coor, mask[None, :, :]], axis=0)
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, self.cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)
            loss_msk = np.stack([mask, mask, mask, np.ones_like(mask)], axis=0)
            return obj, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box)


    def __len__(self):
        return self.num
    
    
    def vis_anno(self, idx):
        obj, obj_id , rgb, out, _, _, pose, c, s, _ = self.__getitem__(idx)
        fig, ax = plt.subplots(2, 3)
        fig.suptitle("{}:{}".format(obj, obj_id))

        rgb = rgb.transpose(1,2,0)*255
        out = out.transpose(1,2,0)
        out[:,:,:3] = self.unnorm_coor(out[:,:,:3], obj_id)
        ax[0][0].imshow(rgb.astype(np.uint8)[:,:,::-1])
        ax[0][1].imshow(out[:,:,-1])
        ax[0][2].imshow(out[:,:,:3])
        
        img_bgr = cv2.imread(self.anno[idx]["rgb_pth"])
        img = img_bgr[:,:,::-1] 
        img_pose_vis = self.vis_pose_vertex(obj_id, pose, img.copy())
        box = self.anno[idx]["box_vis"]
        lefttop_point = [int(box[0]), int(box[1])]
        rightbottom_point = [int(box[0]+box[2]), int(box[1]+box[3])]
        img_box_vis = draw_box(img, lefttop_point, rightbottom_point)
        res = self.cfg.dataiter.out_res
        K = self.cam_K
        r, t = coor2pose1(out[:,:,:3], c, s, res, out[:,:,-1], K)
        R_matrix = cv2.Rodrigues(r, jacobian=0)[0]
        pose_cor = pose.copy()
        pose_cor[:,:3] = R_matrix
        img_pose_coor_vis = self.vis_pose_vertex(obj_id, pose_cor, img.copy())
        
        ax[1][0].imshow(img_pose_coor_vis)
        ax[1][1].imshow(img_pose_vis)
        ax[1][2].imshow(img_box_vis)

        
        plt.show()

    def vis_pose_vertex(self, obj_id, pose, img):
        r_matrix = pose[:,:3]
        t_vector = pose[:,3].T 
        r_vector, _ = cv2.Rodrigues(r_matrix)
        
        obj_x = self.model_info[obj_id]["min_x"] * 2
        obj_y = self.model_info[obj_id]["min_y"] * 2
        obj_z = self.model_info[obj_id]["min_z"] * 2
        obj_dim = [obj_x, obj_y, obj_z]
        return draw_pose_vertex(img, obj_dim, r_vector, t_vector, self.cam_K) 