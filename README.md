# XSens Motion Prediction

## Table of Contents

- [Layout](#project-layout)
- [Workflow](#workflow)
- [Conda Environment](#conda-environment)

## Project Layout

    models/                                                          saved models
    notebooks/                                                     notebooks used during rapid prototyping
    src/                                                                source files for prepping data, training models, running tests, and visualizing outputs
        data/                                                          Matlab code for converting MVNX files to CSV files
        common/                                                    common functions and models used during training
        tests/                                                          source code for running tests
        visualization                                                source code for visualizing model outputs

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
