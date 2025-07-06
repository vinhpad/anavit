export CUDA_VISIBLE_DEVICES=0

if true; then
seeds=(66 111 3 5 7)

for seed in ${seeds[@]}
do
python train_bio.py --data_dir ./dataset/cdr \
--transformer_type bert \
--model_name_or_path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
--train_file train_filter.data \
--dev_file dev_filter.data \
--test_file test_filter.data \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 1 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed $seed \
--num_class 2 \
--segmentation_net swin_unet
done

for seed in ${seeds[@]}
do
python train_bio.py --data_dir ./dataset/cdr \
--transformer_type bert \
--model_name_or_path allenai/scibert_scivocab_cased \
--train_file train_filter.data \
--dev_file dev_filter.data \
--test_file test_filter.data \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 1 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed $seed \
--num_class 2 \
--segmentation_net swin_unet
done
fi