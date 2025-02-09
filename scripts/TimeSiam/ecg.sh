export CUDA_VISIBLE_DEVICES=0
model_name=TCN
pretrain_seq_len=300
mask_rate=0.25
sampling_range=6
lineage_tokens=2
representation_using=avg
root_path=/Users/xiaoyudembp/cmsc/dataset
data_path=chapman

python -u run.py \
    --task_name timesiam \
    --is_training 0 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id $data_path \
    --model $model_name \
    --data $data_path \
    --features M \
    --seq_len $pretrain_seq_len \
    --e_layers 10 \
    --d_layers 1 \
    --d_model 320 \
    --d_ff 256 \
    --n_heads 8 \
    --patch_len 12 \
    --stride 12 \
    --mask_rate $mask_rate \
    --sampling_range $sampling_range \
    --lineage_tokens $lineage_tokens \
    --train_epochs 100 \
    --batch_size 16
