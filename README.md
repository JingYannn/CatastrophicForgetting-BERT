# CatastrophicForgetting-BERT
CF problem in BERT. Support contiual fine-tuning BERT on sequence classification tasks(ColA, MRPC and RTE) and token classification tasks(CoNLL-2003, UD).

## Package
- PyTorch 1.7
- python 3.6.9
- Transformers 4.1.0
- conllu
- seqeval
- tqdm

## Code
- `run.py`: the main code to run the training-evaluation code.

- `processors.py`: the code to process text data.

- `utils.py`: some code including formating time and converting arguments to dict.

- `arguments.py`: some code to parse arguments.

## Run the Code
Please use `run_cl.sh` to fine-tune BERT on several tasks sequentially. The order of tasks and the hyperparameters setting of each task are defined in `order1.json`. The arguments are introduced in `arguments.py`.

## TODO
- fix bug of POS (UD).

- self training.
