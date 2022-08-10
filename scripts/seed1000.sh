SEED=1000
OUTPUT_PATH=/home1/sxy/DialogBERT/output/$SEED

CUDA_VISIBLE_DEVICES=2 python main.py \
--output_path $OUTPUT_PATH \
--seed $SEED