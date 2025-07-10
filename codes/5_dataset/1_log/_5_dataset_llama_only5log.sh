








CUDA_VISIBLE_DEVICES=0 python _5_adaptive_cluster_log_challenge.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/9_llama/llama_layer_list_5_only5log.csv& 
CUDA_VISIBLE_DEVICES=2 python _5_adaptive_cluster_log_easy.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/9_llama/llama_layer_list_5_only5log.csv& 
CUDA_VISIBLE_DEVICES=3 python _5_adaptive_cluster_log_piqa.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/9_llama/llama_layer_list_5_only5log.csv& 
# CUDA_VISIBLE_DEVICES=3 python _5_adaptive_cluster_diff_boolq.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 & 
CUDA_VISIBLE_DEVICES=4 python _5_adaptive_cluster_log_winogrande_aug_1.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/9_llama/llama_layer_list_5_only5log.csv& 
CUDA_VISIBLE_DEVICES=5 python _5_adaptive_cluster_log_hellaswag.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/9_llama/llama_layer_list_5_only5log.csv& 



