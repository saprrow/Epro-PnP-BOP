"""
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: ref.py
@time: 18-10-24 下午9:00
@desc:
"""

import numpy as np
import os

# path
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, '..') 
exp_dir = os.path.join(root_dir, 'exp')
cache_dir = os.path.join(root_dir, 'dataset_cache')

# ply models path
objs_path = "/home/zjw/projects/bproc_dataset/3D_models/epro-pnp-ply"

# camera
im_w = 640
im_h = 480
im_c = (im_h / 2, im_w / 2)
K = np.array([[487.4443064071989, 0.0, 320.6240355910305], [0.0, 487.41225417358606, 241.80599773903214], [0.0, 0.0, 1.0]])

