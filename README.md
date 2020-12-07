# CatastrophicForgetting-BERT
CF problem in BERT. Support contiual fine-tuning BERT on sequence classification tasks(ColA, MRPC and RTE) and token classification tasks(CoNLL-2003, UD).

## Compare the prediction results of CoLA after continual fine-tuning

### Experiment1 : NER(10000) -> CoLA -> MRPC

#### Result

##### Accuracy Matrix

| CoNLL-2003  | CoLA  | MRPC |
| :------------ |:---------------:| -----:|
| 0.9883139      | -0.0207027  | 0.3112745 |
| 0.9802894      | 0.5550197        |   0.4338235 |
| 0.9614164 | 0.3715264        |    0.8578431 |

##### Transfer Matrix

| CoNLL-2003  | CoLA  | MRPC |
| :------------ |:---------------:| -----:|
| 0.      | -0.5757224  | -0.5465686 |
| -0.0080245      | 0.        |   -0.4240196 |
| -0.0268975 | -0.1834933        |    0. |

> Explanation: 
> CoNLL-2003 's label lists: "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC";
> avg. sequence length = 15.797846153846153, select the first 10000 sequences from the train set to do fine-tuning 

- **cnt1** : how many times is each label correctly predicted in the first time? [42568, 836, 286, 1812, 1287, 1258, 682, 1781, 233]

- **cnt2** : how many times is each label correctly predicted in the second time? [42537, 712, 240, 1176, 1168, 1144, 670, 1516, 199]

- **cnt3** : how many times is each label correctly predicted in the first time but uncorrectly in the second time? [107, 140, 50, **637**, 120, 119, 28, 267, 34]

- **cnt4** : how many times is each label correctly predicted in the second time but uncorrectly in the first time? [76, 16, 4, 1, 1, 5, 16, 2, 0]

### Experiment2: CoLA -> MRPC -> NER

#### Result 

##### Accuracy Matrix

| CoLA  | MRPC  | CoNLL-2003 |
| :------------ |:---------------:| -----:|
| 0.5699305      | 0.6838235  | 0.2488363 |
| 0.5528801      | 0.8455882        |   0.3384493 |
| 0.3818496 | 0.7647059        |    0.9872816 |


##### Transfer Matrix

| CoLA  | MRPC  | CoNLL-2003 |
| :------------ |:---------------:| -----:|
| 0.      | -0.1617647  | -0.7384453 |
| -0.0170504      | 0.        |   -0.6488323 |
| -0.1880809 | -0.0808823        |    0. |
 
> CoLA's label list: label 0(ungrammatical), label 1(grammatical)

- **cnt1** : sequences predicted correctly in the first time => 860

- **cnt2** : sequences predicted correctly in the second time => 789

- **cnt3** : sequences predicted correctly in the first time but predicted uncorrectly in the second time (forget) => 118
  - **cnt3_0** : how many times is label '0' are forgot? => **117**
  - **cnt3_1** : how many times is label '1' are forgot? => 1

- **cnt4** : sequences predicted correctly in the second time but predicted uncorrectly in the first time => 47
  - **cnt4_0** : how many times is label '0' are learned? => 0
  - **cnt4_1** : how many times is label '1' are learned? => **47**

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
