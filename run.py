import numpy as np
import os
import time

from arguments import parse_args
from collections import defaultdict
from torch.nn import CrossEntropyLoss
from seqeval.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, AutoConfig, \
	AdamW, get_linear_schedule_with_warmup, glue_compute_metrics

from processors import *
from utils import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def partial_name(args):
	task_type = ""
	for key in args.task_params:
		task_type += str(key) + "_"
	part_name = "{}{}_{}".format(task_type, args.seed, args.num_train_epochs)
	return part_name


def save_model(args, task_num, model):
	part_name = partial_name(args)
	if not os.path.exists(os.path.join(args.output_dir, part_name)):
		os.makedirs(os.path.join(args.output_dir, part_name))

	task_parameters = OrderedDict()
	bert_parameters = model.state_dict().copy()

	for layer, weights in bert_parameters.items():
		if layer == "classifier.weight" or layer == "classifier.bias":
			task_parameters[layer] = weights
	del bert_parameters["classifier.weight"]
	del bert_parameters["classifier.bias"]

	torch.save(bert_parameters, os.path.join(args.output_dir, part_name, "bert_parameters_" + str(task_num) + ".pt"))
	torch.save(task_parameters, os.path.join(args.output_dir, part_name, "task_parameters_" + str(task_num) + ".pt"))


def load_examples(args, task, tokenizer, evaluate=False):
	processor = task_processors[task]()
	logger.info("Creating features from dataset file at %s", args.data_dir)
	label_list = processor.get_labels()

	examples = (
		processor.get_dev_examples(os.path.join(args.data_dir, task)) if evaluate else processor.get_train_examples(
			os.path.join(args.data_dir, task))
	)
	features = convert_examples_to_features(
		examples,
		label_list,
		args.max_seq_length,
		tokenizer,
		task,
	)

	# Convert to Tensors and build dataset
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
	all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
	if task == "ner" or task == "pos":
		all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
		dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
	else:
		all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
		dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_id)
	return dataset


def train(args, train_dataset, all_tasks, task_num, model, models, tokenizer, accuracy_matrix):
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
	t_total = len(train_dataloader) * args.num_train_epochs
	warmup_steps = int(args.warmup_proportion * t_total)

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	# Train~
	logger.info("***** Running training *****")
	logger.info(" Num examples = %d", len(train_dataset))
	logger.info(" Instantaneous batch size per GPU = %d", args.train_batch_size)
	logger.info(" Total optimization steps = %d", t_total)

	global_step = 0
	epochs_trained = 0

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
	)

	if args.eval_during_training:
		loss_value = [[], []]
		prev_accs = defaultdict()
		for i in range(task_num):
			prev_accs[i] = []
		eval_steps = []
		train_length = 0
		iterations = 0

	set_seed(args.seed)  # Added here for reproductibility

	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
		start_time = time.time()
		count = 0
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)

			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
			outputs = model(**inputs)
			loss = outputs[0]  # see doc: https://huggingface.co/transformers/v3.1.0/model_doc/bert.html#bertforsequenceclassification

			loss.backward()

			tr_loss += loss.item()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the exploding gradients problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			optimizer.step()
			scheduler.step()  # Update learning rate
			model.zero_grad()
			global_step += 1

			if args.eval_during_training:
				train_length += args.train_batch_size
				iterations += 1
				avg_loss = tr_loss / iterations
				loss_value[0].append(iterations)
				loss_value[1].append(avg_loss)

				if count == 0:
					count += 1
					eval_steps.append(iterations)
					if task_num >= 1:
						save_model(args, task_num, model)
						part_name = partial_name(args)
						for prev_task_num in range(task_num):
							model.load_state_dict(torch.load(
								os.path.join(args.output_dir, part_name,
											 "task_parameters_" + str(prev_task_num) + ".pt")),
								strict=False
							)
							_, _, prev_acc = evaluate(args, model, all_tasks[prev_task_num], tokenizer, accuracy_matrix,
													  task_num, prev_task_num, False, "Previous Task")
							prev_accs[prev_task_num].append(prev_acc)
					model.load_state_dict(torch.load(
						os.path.join(args.output_dir, part_name, "task_parameters_" + str(task_num) + ".pt")),
						strict=False
					)
				elif count == args.num_eval_steps - 1:
					count = 0
				else:
					count += 1

		logger.info("***** Average training loss: {0:.2f} *****".format(tr_loss / global_step))
		logger.info("***** Training epoch took: {:} *****".format(format_time(time.time() - start_time)))

	# Store data for plotting
	if args.eval_during_training:
		file_name = 'cur_task_' + all_tasks[task_num] + 'baseline.txt'
		with open(file_name, "w") as file:
			arr = []
			arr.append(loss_value[0])
			arr.append(loss_value[1])
			arr.append(eval_steps)
			for i in range(task_num):
				arr.append(prev_accs[i])
			file.write(str(arr))
		file.close()

	results = evaluate(args, model, all_tasks[task_num], tokenizer, accuracy_matrix, task_num, task_num, True,
					   "Current Task")

	save_model(args, task_num, model)

	# Evaluating on all tasks - both forward and backward transfer

	for i in range(len(all_tasks)):
		part_name = partial_name(args)
		# Previous tasks
		if i < task_num:
			models[i][1].load_state_dict(
				torch.load(os.path.join(args.output_dir, part_name, "bert_parameters_" + str(task_num) + ".pt")),
				strict=False)
			models[i][1].load_state_dict(
				torch.load(os.path.join(args.output_dir, part_name, "task_parameters_" + str(i) + ".pt")),
				strict=False)
			results, accuracy_matrix, _ = evaluate(args, model, all_tasks[i], tokenizer, accuracy_matrix, task_num, i,
												   True,
												   "Previous Task (Continual)")

		# Future tasks
		elif (i > task_num):
			models[i][1].load_state_dict(
				torch.load(os.path.join(args.output_dir, part_name, "bert_parameters_" + str(task_num) + ".pt")),
				strict=False)
			models[i][1].load_state_dict(
				torch.load(os.path.join(args.output_dir, part_name, "task_parameters_" + str(i) + ".pt")),
				strict=False)
			results, accuracy_matrix, _ = evaluate(args, model, all_tasks[i], tokenizer, accuracy_matrix, task_num, i,
												   True,
												   "Future Task (Continual)")

	return global_step, tr_loss / global_step, accuracy_matrix


def evaluate(args, model, task, tokenizer, accuracy_matrix, train_task_num, current_task_num, log_matrix, prefix=""):
	eval_dataset = load_examples(args, task, tokenizer, evaluate=True)

	if not os.path.exists(os.path.join(args.output_dir, prefix)):
		os.makedirs(os.path.join(args.output_dir, prefix))

	eval_sampler = RandomSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

	# Eval!
	logger.info("***** Running evaluation:: Task : {}, Prefix : {} *****".format(task, prefix))
	logger.info("  Num examples = %d", len(eval_dataset))
	logger.info("  Batch size = %d", args.eval_batch_size)

	eval_loss = 0.0
	nb_eval_steps = 0
	preds = None
	out_label_ids = None

	for batch in tqdm(eval_dataloader, desc="Evaluating"):
		model.eval()
		batch = tuple(t.to(args.device) for t in batch)

		with torch.no_grad():
			inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
			outputs = model(**inputs)
			tmp_eval_loss, logits = outputs[:2]
			eval_loss += tmp_eval_loss.item()
		nb_eval_steps += 1
		if preds is None:
			preds = logits.detach().cpu().numpy()
			out_label_ids = inputs["labels"].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

	eval_loss = eval_loss / nb_eval_steps
	if task == "ner" or task == "pos":
		results = defaultdict()
		preds = np.argmax(preds, axis=2)
		pad_token_label_id = CrossEntropyLoss().ignore_index

		tags_vals = task_processors[task]().get_labels()
		# label_map = {}
		# for i, label in enumerate(tags_vals):
		# 	label_map[label] = i

		# print(label_map)

		label_map = {i: label for i, label in enumerate(tags_vals)}

		# print(label_map2)

		out_label_list = [[] for _ in range(out_label_ids.shape[0])]
		preds_list = [[] for _ in range(out_label_ids.shape[0])]

		for i in range(out_label_ids.shape[0]):
			for j in range(out_label_ids.shape[1]):
				if out_label_ids[i, j] != pad_token_label_id:
					# print(pad_token_label_id)
					# print(CrossEntropyLoss().ignore_index)
					# print(out_label_ids[i, j])
					# print(out_label_ids[i][j])
					# print(label_map[out_label_ids[i][j]])
					# print(out_label_list[i])
					out_label_list[i].append(label_map[out_label_ids[i][j]])
					preds_list[i].append(label_map[preds[i][j]])

		if task == "ner":
			results = {
				"loss": eval_loss,
				"acc": accuracy_score(out_label_list, preds_list),
				"precision": precision_score(out_label_list, preds_list),
				"recall": recall_score(out_label_list, preds_list),
				"f1": f1_score(out_label_list, preds_list),
				"classification report": classification_report(out_label_list, preds_list),
			}
		elif task == "pos":
			results = {
				"loss": eval_loss,
				"acc": accuracy_score(out_label_list, preds_list),
				"precision": precision_score(out_label_list, preds_list),
				"recall": recall_score(out_label_list, preds_list),
				"f1": f1_score(out_label_list, preds_list),
			}

		result = results["acc"]

	else:
		preds = np.argmax(preds, axis=1)

		results = {}
		result = glue_compute_metrics(task, preds, out_label_ids)
		results.update(result)

		# Log evaluation result for the first task CoLA
		if task == "cola":
			tags_vals = task_processors[task]().get_labels()
			label_map = {}
			for i, label in enumerate(tags_vals):
				label_map[label] = i

			if current_task_num == 0:
				eval_result_file = os.path.join(args.output_dir, "eval_results_" + str(task) + "1" + ".txt")
			else:
				eval_result_file = os.path.join(args.output_dir, "eval_results_" + str(task) + "2" + ".txt")
			with open(eval_result_file, "w") as writer:
				writer.write("index\tlabel\tprediction\n")
				for index, (item1, item2) in enumerate(zip(preds, out_label_ids)):
					item1 = label_map[str(item1)]
					item2 = label_map[str(item2)]
					writer.write(f"{index}\t{item1}\t{item2}\n")
			writer.close()

		if task == 'cola':
			result = result['mcc']
		else:
			result = result['acc']

	logger.info("***** Eval results {} {}*****".format(prefix, task))
	for key in sorted(results.keys()):
		logger.info(" %s = %s", key, str(results[key]))

	if log_matrix:
		accuracy_matrix[train_task_num][current_task_num] = format(result, ".7f")

	return results, accuracy_matrix, result


def main():
	args = parse_args()
	print(args)

	set_seed(args.seed)

	# Prepare tasks
	processors = {}
	label_lists = {}
	num_label_list = {}
	for key in args.task_params:
		processors[key] = task_processors[key]()
		label_lists[key] = task_processors[key]().get_labels()
		num_label_list[key] = len(label_lists[key])

	# Configs
	configs = {}
	for key in args.task_params:
		configs[key] = AutoConfig.from_pretrained(args.model_name_or_path,
												  num_labels=num_label_list[key],
												  finetuning_task=key,
                          id2label={str(i): label for i, label in enumerate(label_lists[key])},
                          label2id={label: i for i, label in enumerate(label_lists[key])},
												  )

	# Tokenizer
	tokenizers = {}
	for key in args.task_params:
		tokenizers[key] = AutoTokenizer.from_pretrained(args.model_name_or_path,
												  do_lower_case=args.do_lower_case,
												  )

	# Continual learning
	n = len(configs)
	accuracy_matrix = np.zeros((n, n))
	transfer_matrix = np.zeros((n, n))

	tasks = list(args.task_params.keys())
	models = []

	# Model
	for key in args.task_params:
		if key == "ner" or key == "pos":
			models.append(
				(key, AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=configs[key])))
		else:
			models.append(
				(key, AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=configs[key])))

	for i in range(n):
		models[i][1].to(args.device)
		save_model(args, i, models[i][1])

	for i in range(n):
		if i > 0:
			part_name = partial_name(args)
			# Always load the BERT parameters of previous model
			models[i][1].load_state_dict(
				torch.load(os.path.join(args.output_dir, part_name, "bert_parameters_" + str(i - 1) + ".pt")),
				strict=False)
		new_args = convert_dict(args.task_params[tasks[i]], args)
		train_dataset = load_examples(args, tasks[i], tokenizers[tasks[i]], evaluate=False)
		global_step, tr_loss, accuracy_matrix = train(
			new_args, train_dataset, tasks, i, models[i][1], models, tokenizers[tasks[i]], accuracy_matrix
		)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

	print()
	print("***** Accuracy Matrix *****")
	print()

	print(accuracy_matrix)

	print()
	print("***** Transfer Matrix *****")
	print("Future Transfer => Upper Triangular Matrix  ||  Backward Transfer => Lower Triangular Matrix")
	print()

	for i in range(n):
		for j in range(n):
			transfer_matrix[j][i] = accuracy_matrix[j][i] - accuracy_matrix[i][i]

	print(transfer_matrix)


if __name__ == '__main__':
	main()
