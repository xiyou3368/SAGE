# SAGE
A tensorflow implementation of self attentive graph embedding (SAGE) in WWW 2019
<p align="center">
  <img width="800" src="sage.jpg">
</p>
<p align="justify">

Details can be found in the paper:
> Semi-Supervised Graph Classification: A Hierarchical Graph Perspective.
> Jia Li, Yu Rong, Hong Cheng, Helen Meng, Wenbing Huang, Junzhou Huang.
> WWW, 2019.
> [[Paper]](https://arxiv.org/pdf/1904.05003.pdf)

# Requirements
python            2.7.15
tensorflow        1.90
numpy             1.15.0
networkx          2.1
scipy             1.1.0
sklearn           0.19.1

# Dataset
proteins

# Model options
```
  --epochs                      INT     Number of epochs.                  Default is 17.
  --weight-decay                FLOAT   Weight decay of Adam.              Defatul is 5*10^-4.
  --gamma                       FLOAT   Regularization parameter.          Default is 0.19.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01.
 ```

# Example
use pretrained model

python train.py

train from scratch


python train.py --train True

# Result
The average accuracy for the pretrained model is 0.80328 for proteins.
