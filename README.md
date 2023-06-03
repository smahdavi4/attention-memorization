# Memorization Capacity of Multi-Head Attention in Transformers

This is the code to "Memorization Capacity of Multi-Head Attention in Transformers".

First, install the required libraries:

```bat
pip3 install -r requirements.txt
```

In order to reproduce the experiments, you need to run the bash file named 'reproduce.sh'.

```bat
bash reproduce.sh
```

Note that each experiment in reproduce.sh is run for only one seed. Running for 6 seeds takes approximately 10 GPU days using a V100 GPU.

For the ViT/BERT/GPT2 analysis, please refer to the notebooks in the `analysis` directory.

## Cite

Please consider citing our paper if you use this code in your research work:

```
@article{
}
```

## Questions/Bugs
Please submit a Github issue or contact smahdavi@ece.ubc.ca if you have any questions or find any bugs.
