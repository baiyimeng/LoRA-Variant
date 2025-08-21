### train ###
# nohup deepspeed --master_port 29400 train_tuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "lora" --sample_ratio 0.2 > /root/autodl-tmp/log/qwen7B_lora_0.2.log &
# nohup deepspeed --master_port 29500 train_xtuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "moelora" --sample_ratio 0.2 > /root/autodl-tmp/log/qwen7B_moelora_0.2.log &
# nohup deepspeed --master_port 29600 train_xtuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "plora" --sample_ratio 0.2 > /root/autodl-tmp/log/qwen7B_plora_0.2.log &
# nohup deepspeed --master_port 29400 train_tuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "lora" --sample_ratio 1.0 > /root/autodl-tmp/log/qwen7B_lora_1.0.log &
# nohup deepspeed --master_port 29500 train_xtuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "moelora" --sample_ratio 1.0 > /root/autodl-tmp/log/qwen7B_moelora_1.0.log &
# nohup deepspeed --master_port 29600 train_xtuning.py --model_name "Qwen2.5-7B-Instruct" --peft_type "plora" --sample_ratio 1.0 > /root/autodl-tmp/log/qwen7B_plora_1.0.log &


### eval ###
# nohup python -u eval_metric_base_vllm.py --model_name "Qwen2.5-7B-Instruct" > /root/autodl-tmp/log/qwen7B_base_metric.log &
# nohup python -u eval_metric_tuning_vllm.py --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_0.2/checkpoint-324" > /root/autodl-tmp/log/qwen7B_lora_0.2_metric.log &
# nohup python -u eval_metric_tuning_vllm.py --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_1.0/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_lora_1.0_metric.log &
# accelerate launch --num_processes 6 eval_metric_xtuning_dp.py --batch_size 64 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/plora_0.2/checkpoint-324" > /root/autodl-tmp/log/qwen7B_plora_0.2_metric.log 2>&1 & disown
# accelerate launch --num_processes 6 eval_metric_xtuning_dp.py --batch_size 64 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/plora_1.0/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_plora_1.0_metric.log 2>&1 & disown
# accelerate launch --num_processes 5 eval_metric_xtuning_dp.py --batch_size 64 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/moelora_0.2_4/checkpoint-324" > /root/autodl-tmp/log/qwen7B_moelora_0.2_4_metric.log 2>&1 & disown
# accelerate launch --num_processes 5 eval_metric_xtuning_dp.py --batch_size 64 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/moelora_1.0_4/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_moelora_1.0_4_metric.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_base_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" > /root/autodl-tmp/log/qwen7B_base_ppl.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_tuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_0.2/checkpoint-324" > /root/autodl-tmp/log/qwen7B_lora_0.2_ppl.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_tuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/lora_1.0/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_lora_1.0_ppl.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_xtuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/plora_0.2/checkpoint-324" > /root/autodl-tmp/log/qwen7B_plora_0.2_ppl.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_xtuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/plora_1.0/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_plora_1.0_ppl.log 2>&1 & disown
# accelerate launch --num_processes 2 eval_ppl_xtuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/moelora_0.2_4/checkpoint-324" > /root/autodl-tmp/log/qwen7B_moelora_0.2_4_ppl.log 2>&1 & disown
# accelerate launch --num_processes 6 eval_ppl_xtuning_dp.py --batch_size 4 --model_name "Qwen2.5-7B-Instruct" --peft_model_id="/root/autodl-tmp/output/Qwen2.5-7B-Instruct/moelora_1.0_4/checkpoint-1617" > /root/autodl-tmp/log/qwen7B_moelora_1.0_4_ppl.log 2>&1 & disown