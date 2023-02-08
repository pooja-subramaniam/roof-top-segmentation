Dida Segmentation Task
==============================

preparing module for dida segmentation task

Project Organization
------------

    ├── README.md          <- The top-level README for basic information about using this project.
    ├── data               <- Not tracked by git so add the data to this folder
    │   └── raw            <- The original, immutable data dump.
    │       └── images      <- Original images as .png files
    │       └── labels      <- Original labels as .png files
    │       └── test_images <- Test images that don't have labels
    │
    ├── models             <- Trained and serialized models, configuration used and model training and validation logs
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering), and a short `-` delimited description,
    │                         e.g.`00-data-exploration`.
    │
    ├── reports            <- Generated analysis.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── documentation.md <- Report on approach and performance
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pipreqs dida-segmentation-challenge`
    │
    ├── environment.yml    <- conda exported environment file
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to create dataset and dataloaders
    │   │
    │   ├── models         <- Scripts to define model classes and training loop
    │   │
    │   ├── visualization  <- Scripts to create training and results visualizations
    │   │
    │   ├── main.py        <- Main script to start to run train-predict-evaluate pipeline
    │   │
    │   ├── predict.py     <- Predict module used in main.py or can be used independently
    │   │
    │   ├── utils.py       <- Utility functions
    │   │
    │   ├── config.yaml    <- Set configurations for running 
    
------------

<p><small>Project adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a> </small></p>

------------

To run training pipeline, cd to src and run main.py:
"python3 main.py"