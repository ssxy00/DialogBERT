SEED=0
dataset=personachat
lr=8e-5

DATA_PATH=/home1/sxy/DialogBERT/datasets/
OUTPUT_PATH=/home1/sxy/DialogBERT/output
GPT2_VOCAB_DIR=/home1/sxy/models/transformers3_gpt2-small

CUDA_VISIBLE_DEVICES=1 python main.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--gpt2_vocab_dir $GPT2_VOCAB_DIR \
--dataset $dataset \
--seed $SEED \
--train_batch_size 64 --grad_accum_steps 1 \
--learning_rate $lr \
--n_epochs 50