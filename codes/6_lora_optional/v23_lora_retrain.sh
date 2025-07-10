CUDA_VISIBLE_DEVICES=0 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/test/ \
    --lora_r 8 --lora_alpha 16 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[10, 25, 11, 26, 12, 9, 19]" &

CUDA_VISIBLE_DEVICES=1 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_1/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[26, 30, 24, 29, 23, 21, 18]" &

CUDA_VISIBLE_DEVICES=2 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_2/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[26, 25, 24, 3, 22, 18, 29]" &

CUDA_VISIBLE_DEVICES=3 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_3/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[27, 3, 23, 24, 26, 20, 22]" &

CUDA_VISIBLE_DEVICES=4 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_4/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[3, 25, 24, 10, 11, 26, 19]" &

wait

    
CUDA_VISIBLE_DEVICES=0 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_5/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[3, 26, 23, 25, 28, 19, 22]" &

CUDA_VISIBLE_DEVICES=1 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_6/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[2, 21, 24, 15, 20, 27, 29]" &

CUDA_VISIBLE_DEVICES=2 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_7/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[2, 21, 27, 20, 15, 7, 28]" &

CUDA_VISIBLE_DEVICES=3 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_8/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[10, 9, 26, 25, 24, 8, 19]" &

CUDA_VISIBLE_DEVICES=4 python v23_lora_retrain.py \
    --base_model meta-llama/Meta-Llama-3.1-8B \
    --data_path yahma/alpaca-cleaned \
    --output_dir lora/v23/layerset_9/ \
    --lora_r 16 --lora_alpha 32 --num_epochs 3 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --skip_layer "[10, 9, 26, 19, 18, 8, 25]" &