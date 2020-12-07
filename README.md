# CatastrophicForgetting-BERT
CF problem in BERT. Support contiual fine-tuning BERT on sequence classification tasks(ColA, MRPC and RTE) and token classification tasks(CoNLL-2003, UD).

## Compare the prediction results of CoLA after continual fine-tuning

### CoLA -> MRPC -> NER

### CoLA: label 0(ungrammatical), label 1(grammatical)

**cnt1** : indexes predicted correctly in the first time => 860

**cnt2** : indexes predicted correctly in the second time => 789

**cnt3** : indexes predicted correctly in the first time but predicted uncorrectly in the second time (forget) => 118
  - **cnt3_0** : how many label '0' are forgot? => 117
  - **cnt3_1** : how many label '1' are forgot? => 1

**cnt4** : indexes predicted correctly in the second time but predicted uncorrectly in the first time => 47
  - **cnt4_0** : how many label '0' are learned? => 0
  - **cnt4_1** : how many label '1' are learned? => 47

## Package
- python 3.6.9
- PyTorch 1.7.0+cu101
- Transformers 4.1.0.dev0
- conllu 4.2.1
- seqeval 1.2.2
- tqdm 4.41.1

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
