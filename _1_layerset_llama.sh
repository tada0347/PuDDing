

CUDA_VISIBLE_DEVICES=0 python codes/4_layerset/_4_sleb_easy_likelihood.py --result_folder result/9_llama/data/4_analysis/arc_easy --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=1 python codes/4_layerset/_4_sleb_challenge_likelihood.py --result_folder result/9_llama/data/4_analysis/arc_challenge --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=2 python codes/4_layerset/_4_sleb_piqa_likelihood.py --result_folder result/9_llama/data/4_analysis/piqa --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=3 python codes/4_layerset/_4_sleb_piqa_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/piqa --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=4 python codes/4_layerset/_4_sleb_easy_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/arc_easy --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=5 python codes/4_layerset/_4_sleb_challenge_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/arc_challenge --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &


wait

CUDA_VISIBLE_DEVICES=0 python codes/4_layerset/_4_sleb_winogrande_likelihood_aug.py --result_folder result/9_llama/data/4_analysis/winogrande --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=1 python codes/4_layerset/_4_sleb_hellaswag_likelihood.py --result_folder result/9_llama/data/4_analysis/hellaswag --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=2 python codes/4_layerset/_4_sleb_winogrande_likelihood.py --result_folder result/9_llama/data/4_analysis/winogrande --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=3 python codes/4_layerset/_4_sleb_boolq_likelihood.py --result_folder result/9_llama/data/4_analysis/boolq --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=4 python codes/4_layerset/_4_sleb_hellaswag_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/hellaswag --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=5 python codes/4_layerset/_4_sleb_winogrande_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/winogrande --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
CUDA_VISIBLE_DEVICES=6 python codes/4_layerset/_4_sleb_boolq_likelihood_diff.py --result_folder result/9_llama/data/4_analysis/boolq --model_name meta-llama/Meta-Llama-3.1-8B --eval_ppl True --num_remove_blocks 7 &
