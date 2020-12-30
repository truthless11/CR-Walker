# CR-Walker

Code implementation for our paper "Bridging the Gap between Conversational Reasoning and Interactive Recommendation". 

you can find our paper at [arxiv](https://arxiv.org/abs/2010.10333).

Cite this paper:

```
@article{ma2020bridging,
      title={Bridging the Gap between Conversational Reasoning and Interactive Recommendation}, 
      author={Wenchang Ma and Ryuichi Takanobu and Minghao Tu and Minlie Huang},
      journal={arXiv preprint arXiv:2010.10333},
      year={2020}
}
```



## Data

- [google link](https://drive.google.com/file/d/1YtJWDZI9ZHPCtVvrE1GiS7Ilb8pTn8KK/view?usp=sharing) to raw data and our model checkpoints. Zipped content:

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

- download and unzip to [your home directory]/CR-Walker/.

## Train

- **For GoRecdial**: 

  ```
  python train_gorecdial.py --option train --model_name <your_model_name>
  ```

- **For Redial**: 

  ```
  python train_redial.py --option train --model_name <your_model_name> --pretrain
  ```

  *We implemented an MIM pretraining stage similar to [KGSF](https://arxiv.org/abs/2007.04032) to accelerate training. 



## Test

- **For GoRecdial**

  ```
  python train_gorecdial.py --option test --model_name gorecdial_128
  ```

- **For Redial**:  

  ```
  python train_redial.py --option test --model_name redial_128
  ```

You can directly evaluate the best model checkpoints for the two datasets that we provided. The results may slightly differ from the paper since we re-trained the model.



## Requirements

python==3.6.10

pytorch==1.4.0

torch_geometric==1.6.0