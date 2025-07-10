
CUDA_VISIBLE_DEVICES=0 python 6_train_BERT_likelihood_MSE.py --epochs 10 --csv_files result/9_llama/data/6_adavanced_tasks/all_log.csv --output_dir result/9_llama/router

# CUDA_VISIBLE_DEVICES=0 python -m v23_eval --router_path result/9_llama/router/10 --result_folder result/9_llama/router/10

