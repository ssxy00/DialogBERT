SEED=0
dataset=multiwoz
DATA_PATH=/home1/sxy/DialogBERT/data/
OUTPUT_PATH=/home1/sxy/DialogBERT/output/$SEED
BERT_PATH=/home1/sxy/models/BERT/bert_base_uncased

CUDA_VISIBLE_DEVICES=2 python main.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--bert_path $BERT_PATH \
--dataset $dataset \
--seed $SEED