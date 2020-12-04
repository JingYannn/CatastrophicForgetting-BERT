import os
import csv
import torch

import logging
from typing import List
from conllu import parse_incr
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
	""" A single training/test example for classification. """

	def __init__(self, guid, text_a, text_b, label=None, labels=None):
		"""
		Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
				sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
				Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
				specified for train and dev examples, but not for test examples.
				Only must be specified for sequence classification tasks.
			labels: (Optional) list. The labels for each word of the sequences.  
				This should be specified for train and dev examples, but not for 
				test examples. Only must be specified for token classification tasks.
		"""

		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label
		self.labels = labels


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label_id=None, label_ids=None):
		"""
		Constructs a InputFeature.

		Args:
		    input_ids: Indices of input sequence tokens in the vocabulary.
		    attention_mask: Mask to avoid performing attention on padding token indices.
		        Mask values selected in ``[0, 1]``:
		        Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
		    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
		    label_id: Indices of label corresponding to the input sequence. Only for sequence 
		    	classification tasks it must be specified.
		    label_ids: Indices of labels corresponding to the input sequence. Only for token 
		    	classification tasks it must be specified.
		"""

		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.token_type_ids = token_type_ids
		self.label_id = label_id
		self.label_ids = label_ids


def convert_examples_to_features(
		examples: List[InputExample],
		label_list: List[str],
		max_seq_length: int,
		tokenizer: PreTrainedTokenizer,
		task: None,
) -> List[InputFeatures]:
	"""Convert one example of classification tasks to features."""

	label_map = {label: i for i, label in enumerate(label_list)}

	features = []

	if task == "ner" or task == "pos":
		# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
		pad_token_label_id = CrossEntropyLoss().ignore_index
		# print(pad_token_label_id)
		for (ex_index, example) in enumerate(examples):
			# print("ex_index:", ex_index)
			tokens = []
			label_ids = []
			for word, label in zip(example.text_a.split(), example.labels):
				word_tokens = tokenizer.tokenize(word)

				# bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
				if len(word_tokens) > 0:
					tokens.extend(word_tokens)
					# Use the real label id for the first token of the word, and padding ids for the remaining tokens
					label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

			# Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
			special_tokens_count = tokenizer.num_special_tokens_to_add()#num_added_tokens()  # num_special_tokens_to_add()
			if len(tokens) > max_seq_length - special_tokens_count:
				tokens = tokens[: (max_seq_length - special_tokens_count)]
				label_ids = label_ids[: (max_seq_length - special_tokens_count)]

			# The convention in BERT is:
			# (a) For sequence pairs:
			#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
			#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
			# (b) For single sequences:
			#  tokens:   [CLS] the dog is hairy . [SEP]
			#  type_ids:   0   0   0   0  0     0   0
			#
			# Where "type_ids" are used to indicate whether this is the first
			# sequence or the second sequence. The embedding vectors for `type=0` and
			# `type=1` were learned during pre-training and are added to the wordpiece
			# embedding vector (and position vector). This is not *strictly* necessary
			# since the [SEP] token unambiguously separates the sequences, but it makes
			# it easier for the model to learn the concept of sequences.
			#
			# For classification tasks, the first vector (corresponding to [CLS]) is
			# used as as the "sentence vector". Note that this only makes sense because
			# the entire model is fine-tuned.
			tokens += ["[SEP]"]
			label_ids += [pad_token_label_id]
			segment_ids = [0] * len(tokens)

			tokens = ["[CLS]"] + tokens
			label_ids = [pad_token_label_id] + label_ids
			segment_ids = [0] + segment_ids

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real
			# tokens are attended to.
			input_mask = [1] * len(input_ids)

			# Zero-pad up to the sequence length.
			padding_length = max_seq_length - len(input_ids)

			input_ids += [0] * padding_length
			input_mask += [0] * padding_length
			segment_ids += [0] * padding_length
			label_ids += [pad_token_label_id] * padding_length

			assert len(input_ids) == max_seq_length
			assert len(input_mask) == max_seq_length
			assert len(segment_ids) == max_seq_length
			assert len(label_ids) == max_seq_length

			if ex_index < 5:
				logger.info("*** Example ***")
				logger.info("guid: %s", example.guid)
				logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
				logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
				logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
				logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
				logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

			features.append(
				InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
							  label_ids=label_ids)
			)
	else:
		for (ex_index, example) in enumerate(examples):

			inputs = tokenizer.encode_plus(
				example.text_a, example.text_b, add_special_tokens=True, max_length=max_seq_length,
				return_token_type_ids=True,
			)
			input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

			attention_mask = [1] * len(input_ids)
			# Zero-pad up to the sequence length.
			padding_length = max_seq_length - len(input_ids)
			input_ids = input_ids + ([0] * padding_length)
			attention_mask = attention_mask + ([0] * padding_length)
			token_type_ids = token_type_ids + ([0] * padding_length)

			assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids),
																							   max_seq_length)
			assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(
				len(attention_mask), max_seq_length
			)
			assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(
				len(token_type_ids), max_seq_length
			)

			label_id = label_map[example.label]

			if ex_index < 5:
				logger.info("*** Example ***")
				logger.info("guid: %s" % (example.guid))
				logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
				logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
				logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
				logger.info("label: %s (id = %d)" % (example.label, label_id))

			features.append(
				InputFeatures(
					input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_id=label_id
				)
			)

	return features


class DataProcessor(object):
	"""Base class for data converters for classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of 'InputExample's for the train set."""
		raise NotImplementedError

	def get_dev_examples(self, data_dir):
		"""Gets a collection of 'InputExample's for the dev set."""
		raise NotImplementedError

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	def read_data_from_file(self, data_dir):
		"""Reads a file."""
		raise NotImplementedError


class NerProcessor(DataProcessor):
	"""Processor for the CoNLL-2003 data set."""

	def read_data_from_file(self, data_dir):
		data = []
		sentence = []
		label = []
		with open(data_dir, encoding="utf-8") as f:
			for line in f:
				if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
					if len(sentence) > 0:
						data.append((sentence, label))
						sentence = []
						label = []
					continue
				splits = line.split(" ")
				sentence.append(splits[0])
				label.append(splits[-1][:-1])

		if len(sentence) > 0:
			data.append((sentence, label))
			sentence = []
			label = []
		return data

	def get_train_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "train.txt")), "train"
		)

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "dev.txt")), "dev"
		)

	def get_test_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "test.txt")), "test"
		)

	def get_labels(self):
		return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC",
				"I-LOC"]  # ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]

	def _create_examples(self, lines, set_type):
		examples = []
		for i, (sentence, labels) in enumerate(lines):
			guid = "%s-%s" % (set_type, i + 1)
			text = ' '.join(sentence)
			labels = labels
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, labels=labels))
		return examples


class PosProcessor(DataProcessor):
	"""Processor for the Universal Dependencies data set."""

	def read_data_from_file(self, data_dir):
		data = []

		with open(data_dir, encoding='utf-8') as f:
			for sentence in parse_incr(f):
				words = []
				labels = []
				for token in sentence:
					words.append(token["form"])
					labels.append(token["upos"])
				assert len(words) == len(labels)

				if len(words) > 0:
					data.append((words, labels))

		return data

	def get_train_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "train.txt")), "train"
		)

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "dev.txt")), "dev"
		)

	def get_test_examples(self, data_dir):
		return self._create_examples(
			self.read_data_from_file(os.path.join(data_dir, "test.txt")), "test"
		)

	def get_labels(self):
		# return ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
		# 		"SCONJ", "SYM", "VERB", "[CLS]", "[SEP]", "X"]
		return ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
				"SCONJ", "SYM", "VERB", "[CLS]", "[SEP]", "_", "X"]

	def _create_examples(self, lines, set_type):
		examples = []
		for i, (sentence, labels) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text = ' '.join(sentence)
			labels = labels
			examples.append(InputExample(guid=guid, text_a=text, text_b=None, labels=labels))
		return examples


class ColaProcessor(DataProcessor):
	"""Processor for the CoLA data set."""

	def read_data_from_file(self, data_dir, quotechar=None):
		"""Read a tab separated value file."""
		with open(data_dir, "r", encoding="utf-8-sig") as f:
			return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			guid = "%s-%s" % (set_type, i)
			text_a = line[3]
			label = line[1]
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples


class MrpcProcessor(DataProcessor):
	"""Processor for the MRPC data set (GLUE version)."""

	def read_data_from_file(self, data_dir, quotechar=None):
		"""Read a tab separated value file."""
		with open(data_dir, "r", encoding="utf-8-sig") as f:
			return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

	def get_train_examples(self, data_dir):
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, i)
			text_a = line[3]
			text_b = line[4]
			label = line[0]
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


class RteProcessor(DataProcessor):
	"""Processor for the RTE data set (GLUE version)."""

	def read_data_from_file(self, data_dir, quotechar=None):
		"""Read a tab separated value file."""
		with open(data_dir, "r", encoding="utf-8-sig") as f:
			return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "train.tsv")), "train")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(self.read_data_from_file(os.path.join(data_dir, "dev.tsv")), "dev")

	def get_labels(self):
		"""See base class."""
		return ["entailment", "not_entailment"]

	def _create_examples(self, lines, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			if i == 0:
				continue
			guid = "%s-%s" % (set_type, line[0])
			text_a = line[1]
			text_b = line[2]
			label = line[-1]
			examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
		return examples


task_processors = {
	"cola": ColaProcessor,
	"mrpc": MrpcProcessor,
	"ner": NerProcessor,
	"pos": PosProcessor,
	"rte": RteProcessor,
}
