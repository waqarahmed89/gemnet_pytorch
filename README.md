# GemNet: Universal Directional Graph Neural Networks for Molecules

Reference implementation in PyTorch of the geometric message passing neural network (GemNet). You can find its original [TensorFlow 2 implementation in another repository](https://github.com/TUM-DAML/gemnet_tf). GemNet is a model for predicting the overall energy and the forces acting on the atoms of a molecule. It was proposed in the paper:

**[GemNet: Universal Directional Graph Neural Networks for Molecules](https://www.in.tum.de/daml/gemnet/)**  
by Johannes Gasteiger, Florian Becker, Stephan Günnemann  
Published at NeurIPS 2021

and further analyzed in

**[How robust are modern graph neural network potentials in long and hot molecular dynamics simulations?](https://www.in.tum.de/daml/gemnet/)**  
by Sina Stocker\*, Johannes Gasteiger\*, Florian Becker, Stephan Günnemann and Johannes T. Margraf  
Published in Machine Learning: Science and Technology, 2022

\*Both authors contributed equally to this research. Note that the author's name has changed from Johannes Klicpera to Johannes Gasteiger.

## Dependencies
### Install cuda 11.7
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu1804-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu1804-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt clean 
sudo apt update 
sudo apt purge nvidia-* 
sudo apt autoremove 
sudo apt install -y cuda
```

### Install cudnn 8
```
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Environment and Packages
```
conda create -y -n gemnet python=3.8
conda activate gemnet
conda install -y ipykernel
ipython kernel install --user --name=gemnet
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt

# optional (if required)
pip install --upgrade protobuf==3.20
pip install PyYAML
```

## Run the code
Adjust config.yaml (or config_seml.yaml) to your needs.
This repository contains notebooks for training the model (`train.ipynb`) and for generating predictions on a molecule loaded from [ASE](https://wiki.fysik.dtu.dk/ase/) (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`). Further, a notebook is provided to show how GemNet can be used for MD simulations (`ase_example.ipynb`).

```
# To run training in background
jupyter nbconvert --to notebook --execute train.ipynb 
```

## Compute scaling factors
You can either use the precomputed scaling_factors (in scaling_factors.json) or compute them yourself by running fit_scaling.py. Scaling factors are used to ensure a consistent scale of activations at initialization. They are the same for all GemNet variants.

## Contact
Please contact j.gasteiger@in.tum.de if you have any questions.

## Cite
Please cite our papers if you use the model or this code in your own work:

```
@inproceedings{gasteiger_gemnet_2021,
  title = {GemNet: Universal Directional Graph Neural Networks for Molecules},
  author = {Gasteiger, Johannes and Becker, Florian and G{\"u}nnemann, Stephan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```

```
@article{stocker_robust_2022,
  title = {How robust are modern graph neural network potentials in long and hot molecular dynamics simulations?},
  author = {Stocker, Sina and Gasteiger, Johannes and Becker, Florian and G{\"u}nnemann, Stephan and Margraf, Johannes T.},
	volume = {3},
	doi = {10.1088/2632-2153/ac9955},
	number = {4},
	journal = {Machine Learning: Science and Technology},
	year = {2022},
	pages = {045010},
}
```
