# geometric-object-grasper
This repository is an implementation of the paper 'A Geometric Approach for Grasping Unknown Objects with Multi-Fingered Hands' in PyBullet.

## Installation
```shell
git clone git@github.com:mkiatos/geometric-object-grasper.git
cd geometric-object-grasper

virtualenv ./venv --python=python3
source ./venv/bin/activate
pip install -e .
```

In this implementation PytTorch 1.9.0 was used:
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Quick Demo
To download the pre-trained models of the shape completion network, run the following commands:
```commandline

```

## Training the Shape Completion Network
<p align="center">
  <img src="images/vae.png" width="700" />
</p>

To collect data for training the Shape Completion network, run the following command:
```commandline

```
Then, to train the network from scratch:
```commandline
python train_shape_net.py --dataset_dir path_to_dataset --epochs 100 --batch_size 4 --lr 0.0001
```

## Evaluation


## Citing
If you find this code useful in your work, please consider citing:
```shell
@article{kiatos2020geometric,
  title={A geometric approach for grasping unknown objects with multifingered hands},
  author={Kiatos, Marios and Malassiotis, Sotiris and Sarantopoulos, Iason},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={3},
  pages={735--746},
  year={2020},
  publisher={IEEE}
}
```