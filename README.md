# (ICLR 2024) Memorization Capacity of Multi-Head Attention in Transformers

This is the official code repository for the paper [Memorization Capacity of Multi-Head Attention in Transformers](https://openreview.net/forum?id=MrR3rMxqqv).

## Reproducing the Experiments

First, install the required libraries:

```bat
pip3 install -r requirements.txt
```

In order to reproduce the experiments, you need to run the bash file named 'reproduce.sh'.

```bat
bash reproduce.sh
```

Note that each experiment in reproduce.sh is run for only one seed. Running for 6 seeds takes approximately 20 GPU days using a V100 GPU.

For the ViT/BERT/GPT2 analysis, please refer to the notebooks in the `analysis` directory.

## Cite

Please consider citing our paper if you use this code in your research work:

```
@inproceedings{
mahdavi2024memorization,
title={Memorization Capacity of Multi-Head Attention in Transformers},
author={Sadegh Mahdavi and Renjie Liao and Christos Thrampoulidis},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=MrR3rMxqqv}
}
```

## Questions/Bugs
Please submit a Github issue or contact smahdavi@ece.ubc.ca if you have any questions or find any bugs.
