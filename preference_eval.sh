#! /bin/bash
ratio=5.0
t=0.5
dz=100
p_id=1

model_name=mistral7b
python \
  preference_eval.py \
  --conv_type llama2 \
  --model_path mistralai/Mistral-7B-Instruct-v0.1 \
  --principle_id $p_id \
  --temperature $t \
  --ratio $ratio \
  --data_size $dz \
  --output_data_file test_outputs/${model_name}_preference_data_${ratio}.json \
  --output_result_file test_outputs/${model_name}_preference_result_${ratio}.json \
  &> test_outputs/log/${model_name}_preference_result_${ratio}.log &
