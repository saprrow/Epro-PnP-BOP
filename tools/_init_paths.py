"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', 'lib')
tools_path = os.path.join(this_dir, '..', 'tools')
add_path(lib_path)
add_path(tools_path)