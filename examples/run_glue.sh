

python run_glue.py \
--data_dir '/root/A/5级单标签分类/train_data' \
--model_type 'bert' \
--model_name_or_path '/root/models/chinese/bert/pytorch/bert-base-chinese' \
--task_name 'sst-2' \
--log_name '5级单标签分类_1' \
--output_dir 'output' \
--max_seq_length 250 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 5e-5 \
--num_train_epochs 8 \
--warmup_steps 100 \
--logging_steps 300 \
--save_steps 300 \
--overwrite_output_dir \
--overwrite_cache \
--do_train  \
--do_eval \
--evaluate_during_training \
--do_lower_case

