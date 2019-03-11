#/bin/bash
#export CUDA_VISIBLE_DEVICES=0 && python train/test.py --gpu 0 --num_point 512 --model frustum_pointnets_v2 --model_path train/avod_car/model.ckpt --output train/detection_results_avod --dump_result
#train/kitti_eval/evaluate_object_3d_offline /data/ssd/public/jlliu/Kitti/object/training/label_2/ train/detection_results_avod
export CUDA_VISIBLE_DEVICES=0 && python train/test.py --gpu 0 --model frustum_pointnets_v2 --model_path train/log/bk/model.ckpt.040 --output test_results --num_point 512 --batch_size 8 --dump_result
train/kitti_eval/evaluate_object_3d_offline /data/ssd/public/jlliu/Kitti/object/training/label_2/ test_results
