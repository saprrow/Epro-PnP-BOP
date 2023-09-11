imgs=$1
python inference.py --cfg exps_cfg/bop/inference.yaml -y ../checkpoints/yolo.pt -s $imgs
