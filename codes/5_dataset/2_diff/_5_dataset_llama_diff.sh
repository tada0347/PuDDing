








CUDA_VISIBLE_DEVICES=0 python _5_adaptive_cluster_diff_challenge.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 
CUDA_VISIBLE_DEVICES=1 python _5_adaptive_cluster_diff_easy.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 
CUDA_VISIBLE_DEVICES=2 python _5_adaptive_cluster_diff_piqa.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 
# CUDA_VISIBLE_DEVICES=3 python _5_adaptive_cluster_diff_boolq.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 
CUDA_VISIBLE_DEVICES=5 python _5_adaptive_cluster_diff_winogrande_aug_1.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 
CUDA_VISIBLE_DEVICES=6 python _5_adaptive_cluster_diff_hellaswag.py --model_name meta-llama/Meta-Llama-3.1-8B --folder result/9_llama/data/3_onlydiff & 



