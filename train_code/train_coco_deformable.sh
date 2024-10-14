GPUS_PER_NODE=$1 ./tools/run_dist_launch.sh $1 ./configs/improved_baseline.sh \
    --batch_size 4 \
    --dataset_file coco \
    --coco_path ./data/coco \
    --name deformable \
    --epochs 1
