# Virginia Tech Natural Motion Processing

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
