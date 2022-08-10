#!/bin/bash
#SBATCH -J dialogbert_seed0         # 作业名
#SBATCH -o dialogbert_seed0.out       # 输出文件，也可删除此行由系统自动指定.
#SBATCH --qos=normal       # qos(quality of service): normal or debug, 对应不同优先级及最大可用时长.
#SBATCH -p gpu             # 指定 partition: cpu, gpu
#SBATCH --cpus-per-task=3  # 申请 cpu processor 数; 可用内存与申请 cpu processor 数成正比.
#SBATCH --gres=gpu:1
#SBATCH -N 1               # 申请节点数
#SBATCH -t 1-00:00:00        # 申请 Job 运行时长0小时5分钟0秒,若要申请一天时间以上,如申请1天,书写格式为#SBATCH -t 1-00:00:00

cd ../

SEED=0
OUTPUT_PATH=/mnt/lustrefs/data/shenxinyi/DialogBERT/output/$SEED

python main.py \
--output_path $OUTPUT_PATH \
--seed $SEED