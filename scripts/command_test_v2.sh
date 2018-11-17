#/bin/bash
export CUDA_VISIBLE_DEVICES=0 && python train/test.py --gpu 0 --num_point 512 --model frustum_pointnets_v2 --model_path train/avod_car/model.ckpt --output train/detection_results_avod --dump_result > test.log 2>&1
train/kitti_eval/evaluate_object_3d_offline /data/ssd/public/jlliu/Kitti/object/training/label_2/ train/detection_results_avod
