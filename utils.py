import random
import datetime
import os
import torch
import numpy as np
from collections import OrderedDict


def format_time(elapsed):
	"""
	Takes a time in seconds and returns a string hh:mm:ss
	"""
	# Round to the nearest second
	elapsed_rounded = int(round(elapsed))

	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


def convert_dict(config, args):
	new_config = AttrDict()
	new_config.update(config)
	new_config.update({
		"num_train_epochs": args.num_train_epochs,
		"seed": args.seed,
		"train_batch_size": args.train_batch_size,
		"eval_batch_size": args.eval_batch_size,
		"warmup_proportion": args.warmup_proportion,
		"model_name_or_path": args.model_name_or_path,
		"device": args.device,
		"output_dir": args.output_dir,
		"data_dir": args.data_dir,
		"max_seq_length": args.max_seq_length,
		"eval_during_training": args.eval_during_training,
		"num_eval_steps": args.num_eval_steps,
		"task_params": args.task_params,
	})
	return new_config
