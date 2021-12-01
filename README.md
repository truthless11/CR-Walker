# CR-Walker

Code for paper "CR-Walker: Conversational Recommender System with Tree-structured Graph Reasoning and Dialog Acts" EMNLP 2021.

you can find our paper at [arxiv](https://arxiv.org/abs/2010.10333).

Cite this paper:

```
@article{ma2021crwalker,
      title={CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation}, 
      author={Wenchang Ma and Ryuichi Takanobu and Minlie Huang},
      journal={arXiv preprint arXiv:2010.10333},
      year={2020}
}
```



## Data

- [google link](https://drive.google.com/drive/folders/1Jg65ibsj_2tybZyCQnGD7y9a80FlCX61?usp=sharing) to raw data and our model checkpoints (generation models will be released soon!). Table of content: 

  ```
  CR-Walker
  ├─data
  │  ├─gorecdial
  │  │  └─raw
  │  ├─gorecdial_gpt
  │  ├─redial
  │  │  └─raw
  │  └─redial_gpt
  └─saved
  ```

- download to [your home directory]/CR-Walker/.

## Train

- **For GoRecdial**: 

  ```
  python train_gorecdial.py --option train --model_name <your_model_name> --pretrain
  ```

- **For Redial**: 

  ```
  python train_redial.py --option train --model_name <your_model_name> --pretrain 
  ```

  We implemented an MIM pretraining stage similar to [KGSF](https://arxiv.org/abs/2007.04032) to accelerate training. Also, we provided option of adding wordnet features by adding "--word_net" as command line option.


## Test Recommendation

- **For GoRecdial**

  ```
  python train_gorecdial.py --option test --model_name gorecdial_reason_128
  ```

- **For Redial**:  

  ```
  python train_redial.py --option test --model_name redial_reason_128
  ```

  You can directly evaluate the best model checkpoints for the two datasets that we provided. The results may slightly differ from the paper since we re-trained the model. Note that the reasoning width (*'sample'* argument in **conf.py**) has been set to 1 for speed during training. You can tune it larger along with the selection threshold (*'threshold'* argument in **conf.py**) to yield better performance.


## Test Generation

- **For GoRecdial**

  ```
  python train_gorecdial.py --option test_gen --model_name gorecdial_reason_128
  ```

- **For Redial**:  

  ```
  python train_redial.py --option test_gen --model_name redial_reason_128
  ```

  Similarly, you can tune the selection threshold, reasoning width and max number of leaf nodes (*'max_leaf'* argument in **conf.py**) to control generation. 


## Requirements

python==3.6.10

pytorch==1.4.0

torch_geometric==1.6.0
