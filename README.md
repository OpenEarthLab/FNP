# FNP: Fourier Neural Processes for Arbitrary-Resolution Data Assimilation

This repo contains the official PyTorch codebase of <a href="https://arxiv.org/abs/2406.01645" target="_blank">FNP</a>. Our paper is accepted by NeurIPS 2024.

## Codebase Structure

- `configs` contains all the experiment configurations.
    - `Adas` contains the configuration to train the <a href="https://arxiv.org/abs/2312.12462" target="_blank">Adas</a> model.
    - `ConvCNP` contains the configuration to train the <a href="https://arxiv.org/abs/1910.13556" target="_blank">ConvCNP</a> model.
    - `FNP` contains the configuration to train the FNP model.
- `data` contains the ERA5 data.
- `datasets` contains the dataset and the mean and standard deviation values of ERA5 data.
- `models` contains the data assimilation models and the forecast model <a href="https://arxiv.org/abs/2304.02948" target="_blank">FengWu</a> (ONNX version).
- `modules` contains the basic modules used for all the data assimilation models.
- `utils` contains the files that support some basic needs.
- `train.py` and `inference.py` provide training and testing pipelines.

We provide the <a href="https://drive.google.com/file/d/1kpzV2vaLbM23g_09AwC_d2hvZFxqVhBT/view?usp=sharing" target="_blank">ONNX model</a> of FengWu with 128×256 resolution for making forecasts. The ERA5 data can be downloaded from the official website of <a href="https://cds.climate.copernicus.eu" target="_blank">Climate Data Store</a>. 

## Setup

First, download and set up the repo

```
git clone https://github.com/OpenEarthLab/FNP.git
cd FNP
```

Then, download and put the ERA5 data and forecast model `FengWu.onnx` into corresponding positions according to the codebase structure.

Deploy the environment given below

```
python version 3.8.18
torch==1.13.1+cu117
```

## Training

We support multi-node and multi-gpu training. You can freely adjust the number of nodes and GPUs in the following commands.

To train the FNP model with the default configuration of `<24h lead time background, 10% observations with 128×256 resolution>`, just run

```
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=29500 train.py
```

You can freely choose the experiment you want to perform by changing the command parameters. For example, if you want to train the `ConvCNP` model with the configuration of `<48h lead time background, 1% observations with 256×512 resolution>`, you can run

```
torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 --master_port=29500 train.py --lead_time=48 --ratio=0.99 --resolution=256 --rundir='./configs/ConvCNP'
```

Please make sure that the parameter `--lead_time` is an integer multiple of 6, because the forecast model has a single-step forecast interval of six hours.

**The resolution and ratio of the observations used for data assimilation can be arbitrary (the original resolution of ERA5 data is 721×1440), which are not limited to the settings given in our paper.**

## Evaluation

The commands for testing are the same as for training. 

For example, you can use 1 GPU on 1 node to evaluate the performance of `Adas` model with the configuration of `<24h lead time background, 10% observations with 721×1440 resolution>` through

```
torchrun --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=29500 inference.py --resolution=721 --rundir='./configs/Adas'
```

The best checkpoint saved during training will be loaded to evaluate the MSE, MAE, and WRMSE metrics for all variables on the testset.

## BibTeX
```bibtex
@article{chen2024fnp,
  title={FNP: Fourier Neural Processes for Arbitrary-Resolution Data Assimilation},
  author={Chen, Kun and Chen, Tao and Ye, Peng and Chen, Hao and Chen, Kang and Han, Tao and Ouyang, Wanli and Bai, Lei},
  journal={arXiv preprint arXiv:2406.01645},
  year={2024}
}
```