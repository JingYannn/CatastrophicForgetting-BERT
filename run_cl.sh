#!/bin/bash
# pip install git+https://github.com/huggingface/transformers.git
# pip install conllu
# pip install seqeval
# wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
# python3 download_glue_data.py --data_dir='data/' --tasks='CoLA,MRPC,RTE'
# mv data/CoLA/ data/cola/
# mv data/MRPC/ data/mrpc/
# mv data/RTE/ data/rte/
python3 run.py \
--data_dir "data" \
--task_params "ner.json" \
--cuda \
--do_lower_case \
--model_name_or_path "bert-base-uncased" \
--output_dir "out" \
--train_batch_size 32 \
--num_train_epochs 1 \
--seed 42