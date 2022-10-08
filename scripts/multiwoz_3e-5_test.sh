SEED=0
dataset=multiwoz
lr=3e-05
ckpt=41

DATA_PATH=/home1/sxy/DialogBERT/datasets/
GPT2_VOCAB_DIR=/home1/sxy/models/transformers3_gpt2-small
RELOAD_PATH=/home1/sxy/DialogBERT/output/${dataset}/lr${lr}/models/checkpoint-${ckpt}
EVAL_OUTPUT_PATH=/home1/sxy/DialogBERT/output/results_new/${dataset}_lr${lr}_ckpt${ckpt}.txt

CUDA_VISIBLE_DEVICES=1 python main.py \
--do_test \
--data_path $DATA_PATH \
--gpt2_vocab_dir $GPT2_VOCAB_DIR \
--dataset $dataset \
--seed $SEED \
--reload_path $RELOAD_PATH \
--eval_output_path $EVAL_OUTPUT_PATH

