# Cross-Entropy is All You Need to Invert the Data Generating Process

Official Python codebase to reproduce the work presented in **Cross-Entropy is All You Need to Inverts the Data Generating Process** [arxiv](https://arxiv.org/abs/2410.21869) by Patrik Reizinger, Alice Bizeul, Attila Juhos, Julia E. Vogt, Randall Balestriero, Wieland Brendel, David Klindt.

## About

This project offers identifiability results for a data generating process in which latents are clustered on a unit hyper-sphere. We show how latent variables can be identified when observing either cluster (_supervised_ setting) or instance index (_self-supervised_ setting).
Empirically we validate our identifiability results on numerical simulations, on the DisLib datasets and ImageNet-X.

<p align="center">
    <img src="https://github.com/klindtlab/csi/blob/main/assets/overview.png" alt="overview" width="200">
</p>

## Code Structure 

```
.
├── assets                       # assets for the README file 
├── simulation                   # numerical simulation (section 4.1)
├── dislib                       # DisLib datasets      (section 4.2)
├── simulation                   # ImageNet-X dataset   (section 4.3)
└── requirements.txt             # installation requirements
```

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/klindtlab/csi.git
cd csi
pip install -r requirements.txt