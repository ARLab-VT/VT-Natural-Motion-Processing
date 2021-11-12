# Virginia Tech Natural Motion Processing

This repository was written to help analyze the Virginia Tech Natural Motion Dataset. The dataset contains 40 hours of unscripted human motion collected in the open world using XSens MVN Link. The dataset, metadata and more information is available through the Virginia Tech University Libraries: https://data.lib.vt.edu/articles/dataset/Virginia_Tech_Natural_Motion_Dataset/14114054/2.

## Table of Contents

- [Layout](#project-layout)
- [Workflow](#workflow)
- [Dependencies](#dependencies)
- [Conda Environment](#conda-environment)

## Project Layout
                                       
    src/                                                                                                  
        common/
        seq2seq/
        transformers                                       
        matlab/

## Dependencies

    numpy==1.18.1
    h5py==2.10.0
    matplotlib==3.1.3
    torch==1.6.0

## Setup

- Clone the repo locally
- Setup the conda environment
    - `$ conda create -n vt-nmp python=3.7`
- Install requirements
    - `$ pip install -r requirements.txt`

## Conda Environment

An Anaconda environment is used to help with development. The environment's main dependency is PyTorch, which will be installed when setting up the workflow above.

## License

Please see the LICENSE for more details. If you use our code or models in your research, please cite our paper:

```
@article{geissinger2020motion,
  title={Motion inference using sparse inertial sensors, self-supervised learning, and a new dataset of unscripted human motion},
  author={Geissinger, Jack H and Asbeck, Alan T},
  journal={Sensors},
  volume={20},
  number={21},
  pages={6330},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
