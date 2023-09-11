import _init_paths

from config import config
from model import build_model
from utils.img import zoom_in
from utils.visual import draw_pose_vertex, draw_box
import cv2
from scipy.stats import truncnorm
import numpy  as np
import ref
import torch
from scipy.spatial.transform import Rotation as R
from ops.pnp.camera import PerspectiveCamera
from ops.pnp.cost_fun import AdaptiveHuberPnPCost
from ops.pnp.levenberg_marquardt import LMSolver
from ops.pnp.epropnp import EProPnP6DoF
from matplotlib import pyplot as plt
from detector import Detector



class PoseModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg 
        self.cam_intrinsic_np = cfg.dataset.camera_matrix.astype(np.float32)
        self.cam_intrinsic = torch.from_numpy(self.cam_intrinsic_np)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no lens distortion
        
        self.epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=3)).cuda(cfg.pytorch.gpu)

        self.model, _ = build_model(cfg)
        self.model.eval()
    

    def pose_inference(self, img_info):
        cfg = self.cfg
        img = img_info["img"]
        box = img_info["box"]
        c, s = self.xywh_to_cs_dzi(box, s_max=max(img.shape[1:]))
        rgb, c_h_, c_w_, s_ = zoom_in(img, c, s, cfg.dataiter.inp_res)
        c = np.array([c_w_, c_h_])
        s = s_
        c = torch.tensor(c).unsqueeze(0)
        s = torch.tensor(s).unsqueeze(0)
        wh_unit = s.to(torch.float32) / cfg.dataiter.out_res 
        rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0)
        
        obj_size = img_info["obj_size"]
        with torch.no_grad():
            (noc, w2d, scale), _ = self.model(rgb_tensor)
        
        x3d, w2d, x2d, valid_mask = self.coors_post_process(noc, w2d, s, c, obj_size, scale)
        pose_init = self.pnp(x3d, x2d, valid_mask) 
        pose = self.pose_refine(x2d, w2d, x3d, pose_init, wh_unit)
        
        return pose


    def xywh_to_cs_dzi(self, xywh, s_max=None, tp='uniform'):
        x, y, w, h = xywh

        c = np.array([x, y]) # [c_w, c_h]
        s = max(w, h)*self.cfg.augment.pad_ratio
        if s_max != None:
            s = min(s, s_max)
        return c, s

    
    def coors_post_process(self, x3d_pre, w2d_pre, s_box, c_box, obj_size, scale):
        bs = x3d_pre.shape[0]
        dim = [[abs(obj_size[0]),
                abs(obj_size[1]),
                abs(obj_size[2])]]
        dim = x3d_pre.new_tensor(dim)  # (n, 3)
        x3d = x3d_pre.permute(0, 2, 3, 1) * dim[:, None, None, :]

        w2d = w2d_pre.flatten(2)
        # we use an alternative to standard softmax, i.e., normalizing the mean before exponential map
        w2d = w2d.softmax(dim=-1).reshape(bs, 2, 64, 64) * scale[..., None, None]
        pred_conf = w2d.mean(dim=1)  # (n, h, w)
        w2d = w2d.permute(0, 2, 3, 1)  # (n, h, w, 2)
        pred_conf_np = pred_conf.cpu().numpy()
        valid_mask = pred_conf_np >= np.quantile(pred_conf_np.reshape(bs, -1), 0.8,
                                                 axis=1, keepdims=True)[..., None]
        out_res = self.cfg.dataiter.out_res
        s = s_box.to(torch.int64)  # (n, )
        wh_begin = c_box.to(torch.int64) - s[:, None] / 2.  # (n, 2)
        wh_unit = s.to(torch.float32) / out_res  # (n, )

        wh_arange = torch.arange(out_res, device=x3d.device, dtype=torch.float32)
        y, x = torch.meshgrid(wh_arange, wh_arange, indexing='ij')  # (h, w)
        # (n, h, w, 2)
        x2d = torch.stack((wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
                           wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]), dim=-1)


        return x3d, w2d, x2d, valid_mask


    def pnp(self, x3d, x2d, mask):
        x2d_np = x2d.cpu().numpy()
        x3d_np = x3d.cpu().numpy()
        _, R_vector, T_vector = cv2.solvePnP(x3d_np[mask], x2d_np[mask], self.cam_intrinsic_np, 
                                             self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        q = R.from_rotvec(R_vector.reshape(-1)).as_quat()[[3, 0, 1, 2]]
        R_quat = x2d.new_tensor(q)
        T_vector = x2d.new_tensor(T_vector.reshape(-1))
        pose = torch.cat((T_vector, R_quat), dim=-1)  # (n, 7)
        
        return pose


    def pose_refine(self, x2d, w2d, x3d, pose_init, wh_unit):
        bs = x2d.shape[0]
        x2d = x2d.reshape(bs, -1, 2)
        w2d = w2d.reshape(bs, -1, 2) / wh_unit[:, None, None]
        x3d = x3d.reshape(bs, -1, 3)
        camera = PerspectiveCamera(cam_mats=self.cam_intrinsic[None].expand(bs, -1, -1), z_min=0.01)
        cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)

        cost_fun.set_param(x2d, w2d)
        pose_opt = self.epropnp(
            x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, fast_mode=True)[0]
        
        return pose_opt



if __name__ == "__main__":
    cfg = config().parse()
    args = cfg.pytorch
    detector = Detector(args.yolo_model)
    pose_infer = PoseModel(cfg)

    img = cv2.imread(args.source)
    img = cv2.resize(img, tuple(args.im_shape))
    img_infos = detector.detect(img)

    img = img_infos[0]["img"]
    img_vis = img[:,:,::-1] 
    for info in img_infos:
        cls = info["cls"]
        info["obj_size"] = np.array(cfg.dataset.model_info[cls]) 

        img_info = info
        pose = pose_infer.pose_inference(img_info)
        
        pose = np.array(pose)
        T_vector = -pose[:3]
        R_vector = R.from_quat(pose[3:][[1,2,3,0]]).as_rotvec()
        
        cam_k = pose_infer.cam_intrinsic_np
        obj_size = img_info["obj_size"]
        
        img = draw_pose_vertex(img_vis, obj_size*2, R_vector, T_vector, cam_k)
        img_vis = img
        
        plt.imshow(img)
        plt.show()
    
    cv2.imwrite("example.png", img[:,:,::-1])