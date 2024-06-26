# COGS 181 Final Project

> [!NOTE]  
> I worked on this project/idea for another class I was enrolled concurrently as of Winter Quarter 2024, ECE 176: Introduction to Deep Learning & Applications, that also had a deep learning final project. https://github.com/kendrick010/ece176_project

## Abstract

Colorization is a widely time-consuming and expensive process for Japanese manga.
Traditionally, Japanese manga are portrayed in black-and-white at debut; however,
a large fraction of manga stall their colorized adaptations due to such bottlenecks.
This project is intended to introduce a system to automate colorization and ideation
for manga-related media using a Generative Adversarial Network (GAN).

## Environment

This project was ran both on Datahub (UCSD) and locally. To install and run this project locally, we recommend setting up a `conda` environment.

```
conda env create -f environment.yml
conda activate COGS181_PROJECT
```

## Dataset

We will be using the [Japanese Manga Dataset](https://www.kaggle.com/datasets/chandlertimm/unified) sourced by CHANDLER TIMM DOLORIEL on `Kaggle`.

To download and setup this dataset, please create a `Kaggle` API token (you must also register for an account if you do not have one). You should get a `kaggle.json` file which contains your `Kaggle` API credentials. Move the `kaggle.json` file into the `/dataset` directory and run `python get_dataset.py`. This download will take a few minutes...

## Getting Started

Run the `get_dataset.py` script, if you havent, to get the managa panel dataset

```
python get_dataset.py
```

Restart and run all cells in the `train_test_split_dataset.ipynb` notebook

Configure the global variables in the `CycleGAN_Manga_Colorization.ipynb` notebook and run.

## Credit

This was adapted from a CycleGANs model that originally mapped horse images to zebra images,
https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca
