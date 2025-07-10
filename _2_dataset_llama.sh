








CUDA_VISIBLE_DEVICES=0 python codes/5_dataset/1_log/_5_adaptive_cluster_log_challenge.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/llama_layer_list_6_advanced_tasks.csv& 
CUDA_VISIBLE_DEVICES=2 python codes/5_dataset/1_log/_5_adaptive_cluster_log_easy.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/llama_layer_list_6_advanced_tasks.csv& 
CUDA_VISIBLE_DEVICES=3 python codes/5_dataset/1_log/_5_adaptive_cluster_log_piqa.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/llama_layer_list_6_advanced_tasks.csv& 
# CUDA_VISIBLE_DEVICES=3 python codes/5_dataset/1_log/_5_adaptive_cluster_diff_boolq.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 & 
CUDA_VISIBLE_DEVICES=4 python codes/5_dataset/1_log/_5_adaptive_cluster_log_winogrande_aug_1.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/llama_layer_list_6_advanced_tasks.csv& 
CUDA_VISIBLE_DEVICES=5 python codes/5_dataset/1_log/_5_adaptive_cluster_log_hellaswag.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/5_log5 --open_path codes/llama_layer_list_6_advanced_tasks.csv& 



