
# 网民情感分类
#python run_glue.py \
#--data_dir '/root/NLP语料/DF比赛/疫情期间网民情绪识别' \
#--model_type 'bert' \
#--model_name_or_path '/home/xyf/models/chinese/bert/pytorch/bert-base-chinese' \
#--task_name 'sst-2' \
#--output_dir 'output_transformers_网民情感分类' \
#--max_seq_length 200 \
#--do_train \
#--do_eval \
#--per_gpu_train_batch_size  24 \
#--per_gpu_eval_batch_size 24 \
#--gradient_accumulation_steps 1 \
#--learning_rate 5e-5 \
#--num_train_epochs 4 \
#--warmup_steps  100 \
#--logging_steps 2000 \
#--save_steps  2000




# 酒店评价分类
python run_glue.py \
--data_dir '/home/xyf/桌面/Disk/NLP语料/文本分类/ChnSentiCorp/ChnSentiCorp情感分析酒店评论' \
--model_type 'bert' \
--model_name_or_path '/home/xyf/models/chinese/bert/pytorch/bert-base-chinese' \
--task_name 'sst-2' \
--output_dir 'output_transformers_酒店情分类' \
--max_seq_length 150 \
--do_train \
--do_eval \
--per_gpu_train_batch_size  8 \
--per_gpu_eval_batch_size 8 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-5 \
--num_train_epochs 4 \
--warmup_steps  10 \
--logging_steps 200 \
--save_steps  200
