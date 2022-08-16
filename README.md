Neural Variational Dropout Processes
====================================

This repository implements the models and algorithms necessary to reproduce the experiments presented in the conference paper `Neural Variational Dropout Processes, Jeon, et al.` ICLR, 2022. <https://openreview.net/forum?id=lyLVzukXi08>

It includes code for running few-shot regression experiments with Gaussian Process with the random kernel (learned variance) as well as for reproducing the 2d image inpainting (MNIST, CelebA, and Omniglot) experiments.

The major component of this repository is as follows:

* ``demo``: contains a simple demo of learning trigonometry data.
  - ``NVDP_demo_TrigData.ipynb``: script to run the demo experiment.
* ``gp``: contains the few-shot regression experiment on Gaussian Process with the random kernel (learned variance).
  - ``main_gp.py``: script to run few-shot GP regression experiment.
* ``2d``: contains the few-shot image inpainting (MNIST, CelebA, and Omniglot) experiments.
  - ``main_2d.py``: script to run few-shot image inpainting experiment.


Dependencies
------------
This code requires the following:

* python 3
* PyTorch 1.0+
* tensorboardX

Data
----
For Gaussian Process and Trigonometry and dataset, they are implemented and included as library files.
For MNIST dataset, the data can be downloaded automatically by the torchvision library.
For CelebA, and Omniglot dataset, the data can be downloaded from <https://tinyurl.com/yjpyxj59>. 
Please download each file, unzip and locate it in the `./data` folder. 
Please see `main_2d.py` and `./data/data_independent.py` for more specific data location.

Usage
-----

* To run the demo experiment, follows the instructions in ``NVDP_demo_TrigData.ipynb``.
* To run the few-shot regression with the Gaussian Process experiment, type the following command on the Linux console: ``python main_gp.py``.
* To run the 2d image inpainting experiment, type the following command on the Linux console: ``python main_2d.py``.

See more optional parameter settings (e.g., changing dataset, models) in each main python file.
This script also includes the implementation of (Conditional) Neural processes <https://github.com/deepmind/neural-processes> as baselines.

To see the results of each experiment, enter into the folder ``./runs/[dataset_name]/[experiment_name]/events/``
and execute tensorboard with the following command: `tensorboard --logdir=./ --port=8888 --samples_per_plugin image=100 --reload_multifile=True --reload_interval 30 --host=0.0.0.0`


Contact
-------
To ask questions or report issues, please open an issue on the issues tracker.


Citation
--------

If you use this code for your research, please cite our paper <https://openreview.net/forum?id=lyLVzukXi08>:
::

  @inproceedings{jeon2022neural,
    title={Neural variational dropout processes},
    author={Jeon, Insu and Park, Youngjin and Kim, Gunhee},
    booktitle={International Conference on Learning Representations},
    year={2022}
  }
 