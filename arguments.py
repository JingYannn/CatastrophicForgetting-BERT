import argparse
import json
import torch

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir",
						type=str,
						default="../data",
						required=True,
						)
	parser.add_argument("--task_params",
						type=str,
						required=True,
						help="JSON file path"
						)
	parser.add_argument("--output_dir",
						type=str,
						default="./output",
						required=True,
						)
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Batch size during training",
						)
	parser.add_argument("--eval_batch_size",
						default=32,
						type=int,
						help="Batch size during evaluating",
						)
	parser.add_argument("--cuda",
						action="store_true",
						)
	# parser.add_argument("--tokenizer_name",
	# 					type=str,
	# 					required=True,
	# 					)
	parser.add_argument("--do_lower_case",
						action="store_true",
						)
	parser.add_argument("--model_name_or_path",
						default="bert-base-uncased",
						type=str,
						required=True,
						)
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						)
	parser.add_argument("--num_train_epochs",
						default=1,
						type=int,
						)
	parser.add_argument("--seed",
						default=42,
						type=int,
						)
	parser.add_argument("--eval_during_training",
						action="store_true",
						help="Evaluation is done and logged every eval_steps"
						)
	parser.add_argument("--num_eval_steps",
						default=10,
						type=int,
						help="Number of training steps to do evaluation",
						)
	parser.add_argument("--warmup_proportion",
						default=0.1,
						type=int,
						)

	args = parser.parse_args()
	with open(args.task_params) as file:
		args.task_params = json.load(file)
	if args.cuda:
		args.device = torch.device("cuda")
	else:
		args.device = torch.device("cpu")
	return args