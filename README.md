# XSens Motion Prediction

## Table of Contents

- [Layout](#project-layout)
- [Workflow](#workflow)
- [Tools](#tools)
- [Python VEnv](#python-virtual-environment)
- [Research](#research)

## Project Layout

    joint-angle-prediction/                                           source code for joint angle prediction using PyTorch
    kinematic-modeling/                                               source code for kinematic modeling using PyTorch
    common/                                                           common files used by main parts of the project

## Workflow

- Clone the repo locally
- Setup the conda environment
    - `$ conda create --name <myenv> --file environment.txt`
- Feature branches will be used for development
    - `$ git checkout -b <descriptive-name>`
- Make changes
    - `$ git add <file-name>`
    - `$ git commit -m "changes to file-name for new feature"`
- Once the feature branch is complete and ready for a pull request, it can be pushed to the remote repository to create a pull request
    - `$ git push origin <descriptive-name>`

## Conda Environment

An Anaconda environment is used to help with development. The environment's main dependency is PyTorch, which will be installed when setting up the workflow above.
