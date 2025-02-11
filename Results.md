# Experimental Results


## environment
```
conda create -n glcm python==3.6.8
conda activate glcm
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install transformers==3.0.2
python -m pip install nltk==3.5
python -m pip install tables
python -m pip install tensorboard
```

## Results on DailyDialog
Run with different random seed:
+ show logs in tensorboard

```
# on ubuntu
cd /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/logs
conda activate glcm
tensorboard --logdir ./
```

+ choose ckpt according to valid loss

| seed | ckpt | valid loss |
|------|------|------------|
| 0    | 56k  | 3.046      |
| 42   | 58k  | 3.039      |
| 1000 | 68k  | 3.044      |

+ evaluate on test set

```
# seed 0
python main.py --do_test --dataset dailydial --reload_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/models/seed0/checkpoint-56000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/results/seed0_ckpt56k.txt
# seed 42
python main.py --do_test --dataset dailydial --reload_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/models/seed42/checkpoint-58000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/results/seed42_ckpt58k.txt
# seed 1000
python main.py --do_test --dataset dailydial --reload_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/models/seed1000/checkpoint-68000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/dailydial/DialogBERT/tiny/results/seed1000_ckpt68k.txt
```

| seed | avg_len | bleu   | meteor | nist   | ppl     | rougeL | loss   |
|------|---------|--------|--------|--------|---------|--------|--------|
| 0    | 15.5718 | 0.2001 | 0.0693 | 0.2077 | 21.8936 | 0.0845 | 3.0862 |
| 42   | 14.9424 | 0.2054 | 0.0703 | 0.1986 | 21.6913 | 0.0859 | 3.0769 |
| 1000 | 15.1806 | 0.2016 | 0.0691 | 0.2007 | 21.7261 | 0.0846 | 3.0785 |
| avg  | 15.2316 | 0.2024 | 0.0696 | 0.2023 | 21.7703 | 0.085  | 3.0805 |


## Results on multiwoz
Run with different random seed:
+ show logs in tensorboard

```
# on ubuntu
cd /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/logs
conda activate glcm
tensorboard --logdir ./
```

+ choose ckpt according to valid loss

| seed | ckpt | valid loss |
|------|------|------------|
| 0    | 76k  | 1.614      |
| 42   | 80k  | 1.613      |
| 1000 | 80k  | 1.613      |


```
# seed 0
python main.py --do_test --dataset multiwoz --reload_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/models/seed0/checkpoint-76000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/results/seed0_ckpt76k.txt
# seed 42
python main.py --do_test --dataset multiwoz --reload_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/models/seed42/checkpoint-80000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/results/seed42_ckpt80k.txt
# seed 1000
python main.py --do_test --dataset multiwoz --reload_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/models/seed1000/checkpoint-80000 \
--eval_output_path /home/ssxy/research/project_data/DialogBERT/output/multiwoz/DialogBERT/tiny/results/seed1000_ckpt80k.txt
```


| seed | avg_len | bleu   | meteor | nist   | ppl    | rougeL | loss   |
|------|---------|--------|--------|--------|--------|--------|--------|
| 0    | 17.0671 | 0.2807 | 0.1708 | 0.5282 | 5.0290 | 0.1815 | 1.6152 |
| 42   | 16.9713 | 0.2790 | 0.1703 | 0.5272 | 5.0080 | 0.1807 | 1.6110 |
| 1000 | 16.9868 | 0.2755 | 0.1679 | 0.5196 | 5.0167 | 0.1801 | 1.6128 |
| avg  | 17.0084 | 0.2784 | 0.1697 | 0.525  | 5.0179 | 0.1808 | 1.6130 |


