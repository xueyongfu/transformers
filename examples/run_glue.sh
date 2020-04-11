

python run_glue.py \
--data_dir '/root/NLP语料/ChnSentiCorp情感分析酒店评论' \
--model_type 'bert' \
--model_name_or_path '/root/models/chinese/bert/pytorch/bert-base-chinese' \
--task_name 'sst-2' \
--log_name '分类' \
--output_dir 'output' \
--max_seq_length 150 \
--do_train  \
--do_eval \
--evaluate_during_training \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--num_train_epochs 5 \
--warmup_steps 100 \
--overwrite_output_dir \
--overwrite_cache