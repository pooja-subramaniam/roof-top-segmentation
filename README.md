Dida Segmentation Task
==============================

preparing module for dida segmentation task

Project Organization
------------

    ├── README.md          <- The top-level README for basic information about using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │    
    ├── models             <- Trained and serialized models, model predictions, or model evaluation logs
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), and a short `-` delimited description,
    │                         e.g.`00-data-exploration`.
    │
    ├── reports            <- Generated analysis.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── documentation.md <- Report on approach and performance
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pipreqs dida_task_repo`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to extract and clean data
    │   │
    │   ├── models         <- Scripts to define model classes
    │   │
    │   └── visualization  <- Scripts to create training and results visualizations
    │   │
    │   ├── main.py        <- Main script to start run to train-predict-evaluate pipeline
    │   ├── train.py       <- Train module used in main.py
    │   ├── predict.py     <- Predict module used in main.py or can be used independently
    │   │
    │   ├── utils.py       <- Utility functions
    │   ├── config.yaml    <- Set configurations for running 
    
------------

<p><small>Project adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a> </small></p>
