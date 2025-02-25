# Cross-Entropy is All You Need to Invert the Data Generating Process

Official Python codebase to reproduce the work presented in **Cross-Entropy is All You Need to Inverts the Data Generating Process** [[arxiv]](https://arxiv.org/abs/2410.21869) by Patrik Reizinger, Alice Bizeul, Attila Juhos, Julia E. Vogt, Randall Balestriero, Wieland Brendel, David Klindt.

## About

This project offers identifiability results for a data generating process in which latents are clustered on a unit hyper-sphere. We show how latent variables can be identified when observing either cluster (_supervised_ setting) or instance (_self-supervised_ setting) index.
Empirically we validate our findings on numerical simulations, DisLib datasets and the ImageNet-X dataset.

<p align="center">
    <img src="https://github.com/klindtlab/csi/blob/main/assets/overview.png" alt="overview" width="800">
</p>

## Code Structure 

```
.
├── assets                       # assets for the README file 
├── simulation                   # numerical simulation (section 4.1)
│   ├── scripts                  #   bash scripts to launch training
│   ├── scripts                  #   package
│   ├── main_sup.py              #   entrypoint to launch supervised learning experiments (table 3)
│   └── main_diet.py             #   entrypoint to launch self-supervised learning experiments (table 2)
├── dislib                       # DisLib datasets      (section 4.2)
├── simulation                   # ImageNet-X dataset   (section 4.3)
└── requirements.txt             # installation requirements
```
Please note that the code for numerical simulations is adapted from the [CL-ICA](https://github.com/brendel-group/cl-ica) repository.


## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/klindtlab/csi.git
cd csi
pip install -r requirements.txt