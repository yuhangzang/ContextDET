GPUS_PER_NODE=$1 ./tools/run_dist_launch.sh $1 ./configs/improved_baseline.sh \
    --batch_size 4 \
    --coco_path ./data \
    --name deformable \
    --dataset_file refcoco \
    --image_set val \
    --eval_type referring \
    --epochs 1 \
    --llm_name opt