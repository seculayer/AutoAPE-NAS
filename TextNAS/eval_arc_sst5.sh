# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1

fixed_arc="$fixed_arc 0 5"
fixed_arc="$fixed_arc 1 7 0"
fixed_arc="$fixed_arc 1 3 0 0"
fixed_arc="$fixed_arc 3 6 0 1 1"
fixed_arc="$fixed_arc 1 1 0 1 0 0"
fixed_arc="$fixed_arc 0 1 0 0 1 0 1"
fixed_arc="$fixed_arc 0 0 0 0 1 1 1 1"
fixed_arc="$fixed_arc 4 6 0 1 1 0 0 0 0"
fixed_arc="$fixed_arc 3 5 0 1 0 1 1 0 1 1"
fixed_arc="$fixed_arc 2 5 0 0 0 0 0 1 0 1 1"
fixed_arc="$fixed_arc 4 7 1 1 0 0 1 0 0 1 1 0"
fixed_arc="$fixed_arc 4 4 1 0 0 1 1 1 1 0 0 0 0"
fixed_arc="$fixed_arc 1 1 1 1 0 1 0 1 0 0 1 0 0 0"
fixed_arc="$fixed_arc 1 2 1 1 0 0 0 0 1 0 0 1 1 0 1"
fixed_arc="$fixed_arc 1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0"
fixed_arc="$fixed_arc 0 3 1 0 0 0 0 0 1 0 0 0 1 1 0 0 1"
fixed_arc="$fixed_arc 1 2 0 0 0 0 0 0 0 0 0 1 1 0 1 0 1 0"
fixed_arc="$fixed_arc 0 4 0 1 1 1 1 0 0 1 0 1 0 0 0 0 1 0 0"
fixed_arc="$fixed_arc 2 3 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1"
fixed_arc="$fixed_arc 2 3 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0"
fixed_arc="$fixed_arc 3 5 1 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 1 1 0 0"
fixed_arc="$fixed_arc 4 7 0 1 1 1 0 0 0 1 0 1 1 1 1 0 1 0 0 0 0 0 0"
fixed_arc="$fixed_arc 0 7 0 0 1 1 1 1 0 1 0 1 1 0 0 0 0 1 0 0 0 0 0 1"
fixed_arc="$fixed_arc 1 3 1 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0"

python main.py \
  --train_ratio=1.0 \
  --valid_ratio=1.0 \
  --min_count=1 \
  --is_mask \
  --embedding_model="glove" \
  --child_lr_decay_scheme="cosine" \
  --data_path="./data/sst" \
  --embedding_path="./data/glove.840B.300d.txt" \
  --class_num=5 \
  --global_seed=1234 \
  --child_optim_algo="adam" \
  --output_dir="./output/output_sst5" \
  --max_input_length=64 \
  --batch_size=256 \
  --eval_batch_size=128 \
  --num_epochs=10 \
  --log_every=50 \
  --eval_every_epochs=1 \
  --child_num_layers=24 \
  --child_out_filters=256 \
  --child_l2_reg=1e-6 \
  --cnn_keep_prob=0.8 \
  --final_output_keep_prob=1.0 \
  --embed_keep_prob=0.8 \
  --lstm_out_keep_prob=0.8 \
  --attention_keep_prob=0.8 \
  --child_lr=0.02 \
  --child_lr_max=0.002 \
  --child_lr_min=0.0002 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --is_cuda \
  --multi_path \
  --child_fixed_arc="${fixed_arc}" \
  --fixed_seed \
  --all_layer_output \
  --output_linear_combine \
  "$@"
