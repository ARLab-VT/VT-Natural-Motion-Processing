# Virginia Tech Natural Motion Processing

## Table of Contents

- [Layout](#project-layout)
- [Workflow](#workflow)
- [Conda Environment](#conda-environment)

## Project Layout
                                       
    src/                                                                                                  
        common/
        seq2seq/
        transformers                                       
        matlab/

## Workflow

- Clone the repo locally
- Setup the conda environment
    - `$ conda env create -f environment.yml`
- Feature branches will be used for development
    - `$ git checkout -b <descriptive-name>`
- Make changes
    - `$ git add <file-name>`
    - `$ git commit -m "changes to file-name for new feature"`
- Once the feature branch is complete and ready for a pull request, it can be pushed to the remote repository to create a pull request
    - `$ git push origin <descriptive-name>`

## Conda Environment

An Anaconda environment is used to help with development. The environment's main dependency is PyTorch, which will be installed when setting up the workflow above.
