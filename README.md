


# SimpleTOD: A Simple Language Model for Task-Oriented Dialogue
Authors: [Ehsan Hosseini-Asl](https://scholar.google.com/citations?user=I9w3ON4AAAAJ&hl=en), [Bryan McCann](https://bmccann.github.io/), [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Semih Yavuz](https://scholar.google.co.uk/citations?user=krh3p8AAAAAJ&hl=en), and [Richard Socher](https://www.socher.org/)

![SimpleTOD](/images/simpletod_autoregressive.gif)

![SimpleTOD single turn](/images/simpletod_single_turn.gif)

## Introduction
Task-oriented dialogue (TOD) systems accomplish a goal described 
by a user in natural language. They often use a pipeline approach. 
Such approach requires natural language understanding (NLU) for belief state tracking, 
dialogue management (DM) for deciding which actions to take based on those beliefs, 
and natural language generation (NLG) for generating responses.

We propose recasting task-oriented dialogue as a simple, causal (unidirectional) 
language modeling task. We show that such an approach can solve all the sub-tasks 
in a unified way using multi-task maximum likelihood training. 
The proposed Simple Task-Oriented Dialogue (SimpleTOD) approach enables modeling of 
the inherent dependencies between the sub-tasks of task-oriented dialogue, 
by optimizing for all tasks in an end-to-end manner.


Paper link: https://arxiv.org/abs/2005.00796

Blog link: https://blog.einstein.ai/simpletod
 

## Table of Contents
- [Installation](#installation) 
- [Usage](#usage) 
    - [Preprocessing](#preprocessing)
    - [DST training](#dst-training)
    - [End-to-End training](#end-to-end-training)
    - [Generation](#generation)
    - [Evaluation](#evaluation)
        - [DST Evaluation](#dst-evaluation)
        - [End-to-End Evaluation](#end-to-end-evaluation)
    - [Demo](#demo)
- [Citation](#citation)
- [License](#license)
 

## Installation

The package general requirements are

- Python >= 3.6
- Pytorch >= 1.2 (installation instructions [here](https://pytorch.org/))
- Transformers >= 2.5.1 (installation instructions [here](https://huggingface.co/transformers/))
 
1- The package can be installed by running the following command.  

```pip install -r requirements.txt```

2- Running inside docker container
```
docker build -t <image_name>:<tag> -f Dockerfile
```

## Usage
This section explains steps to preprocess MultiWOZ dataset and training the model. 

### Preprocessing: 
It includes downloading MultiWOZ dataset, performing delexicaliztion, and creating dataset for language model
```
create_dataset.sh
```
Each dialogue turn will be represented as a sequence, which contains previous user/system turns, belief, action, and delexicalized response

```
<|endoftext|> <|context|> <|user|> i am looking for a college type attraction . <|system|> there are 18 colleges i have found , would you prefer 1 in town centre or in the west ? <|user|> i would like to visit on in town centre please . <|system|> sure , we have thirteen options , 10 of which are free . may i suggest king s college , or hughes hall ? <|user|> okay , may i have their postcode , entrance fee , and phone number ?<|endofcontext|> 
<|belief|> attraction type college , attraction name kings college|hughes hall , attraction area centre <|endofbelief|> 
<|action|> attraction inform name , attraction inform fee , attraction inform post , attraction inform phone <|endofaction|> 
<|response|> sure , the post code to [attraction_name] is [attraction_postcode] , the entrance fee is free , and phone number [attraction_phone] <|endofresponse|> <|endoftext|>
```


### DST training: 
training the model for predicting belief states.   
 
```
train_dst.sh $GPU gpt2 $GPT2_TYPE $BATCH
```

For this task, we include ```none``` slot values in the sequence. 
We observed that this will improve SimpleTOD performance on DST by reducing false positive rates. 
```
<|endoftext|> <|context|> <|user|> am looking for a place to to stay that has cheap price range it should be in a type of hotel <|endofcontext|> 
<|belief|> hotel name not mentioned , hotel area not mentioned , hotel parking not mentioned , hotel pricerange cheap , hotel stars not mentioned , hotel internet not mentioned , hotel type hotel <|endofbelief|> <|endoftext|>
```


### End-to-End training:
In this step, we train SimpleTOD on the sequence of context+belief+action+delex response. 
Compared to DST task, we do not include ```none``` slot values, because of the sequence length limitaiton od GPT2. 
```
train_end2end.sh $GPU gpt2 $GPT2_TYPE $BATCH
```

 
### Generation:

This script will generate SimpeTOD belief/action/responses. 
Generation is based on each dialogue, where it create context for each turn and save the generated belief, action, and responses for the dialogue.

```
CUDA_VISIBLE_DEVICES=$GPU python generate_dialogue.py $CHECKPOINT $DECODING
```
It will save the model output in a json file ```MODEL_OUTPUT``` which contains all dialogues with groundtruth user and system responses as well.
- In order to use DB search during generation, set ```--use_db_search``` (this will use *oracle* DB search results)
- In order to use DB search dynamically, set ```--use_db_search``` and ```--use_dynamic_db```
- To use oracle belief and actions, simple set ```--use_oracle_belief``` and ```--use_oracle_action```

### Evaluation
MultiWOZ evaluation contains two part, Dialogue State Tracking (DST) and End-to-End.  

#### DST evaluation

In order to compute joint accuracy, simply run the following script using the generated
```MODEL_OUTPUT``` file. it will use the generated belief states to compute the metric. It will compute joint accuracy without any label cleaning.
```
python compute_joint_acc.py $MODEL_OUTPUT 
```
There are two types of label cleaning that can be used to compute joint accuracy. 
- To use default lable cleaning suggested by MultiWOZ author, please set ```--default_cleaning``` (for more details, please refer to [MultiWOZ](https://github.com/budzianowski/multiwoz) FAQ.5)
- We found other type of noisy annotation. Please refer to the paper for more details different types of noisy annotations. Here, we provide an option to compute joint accuracy by fixing Type 2 noisy annotation (where one or more slots are not labeled in some turns.) by setting ```--type2_cleaning``` 
- The complete list of Type 2 noisy annotations is [here](noisy_annotations/type_2_noisy_annotations.json). For more details on noisy annotation on MultiWOZ dataset, please refer to the paper


#### End-to-End evaluation

In order to compute inform/success/BLEU, simply run the following script. It will load generated belief states and responses, and computes the metrics. 
```
python evaluate_multiwoz.py $MODEL_OUTPUT
```

### Demo  

In order to test the model in real conversation with human, we have provided a simple script where user can input text in a multi turn setting, and see the responses from SimpleTOD. 
It will generate lexicalized responses and belief states at each turn. For more information, please read the blog.  
```
python demo.py $CHECKPOINT $DECODING
```


## Citation
```
@article{hosseini2020simple,
  title={A simple language model for task-oriented dialogue},
  author={Hosseini-Asl, Ehsan and McCann, Bryan and Wu, Chien-Sheng and Yavuz, Semih and Socher, Richard},
  journal={arXiv preprint arXiv:2005.00796},
  year={2020}
}
```


## License
The code is released under the BSD-3 License - see [LICENSE](LICENSE.txt) for details
