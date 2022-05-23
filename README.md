# 3d-medical-ssl

## Requirements

An optional first step to create a python environment for the project
```
conda create --name 3d-medical-ssl python=3.8
conda activate 3d-medical-ssl
```

Install [torch](https://pytorch.org/) compatible with your CUDA version
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch  # for CUDA >= 11.3
```

Then, install our project
```
git clone https://github.com/V-Soboleva/3d-medical-ssl
cd 3d-medical-ssl
pip install -e .
```

## Training

To train the model, run
```
python train.py --input-data <path_to_data>
```

