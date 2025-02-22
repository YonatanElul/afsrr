# Atrial Fibrillation Screening During Sinus-Rhythm via Analysis of Cardiac Dynamics 
This is an official implementation of the code used in the paper "Atrial Fibrillation Screening During Sinus-Rhythm via Analysis of Cardiac Dynamics" - Yonatan Elul, Noam Keidar, Yael Drori, Alex M. Bronstein, Assaf Schuster, Yael Yaniv.

## System Requirements
The package has been tested on Windows 10, Ubuntu 20.04.2 LTS, and macOS Sequoia 15.0.1 (24A348).

The package requires Python 3.9.6 and the following packages:
* numpy >= 1.26.2
* scipy >= 1.13.1 
* scikit-learn >= 1.3.2 
* wfdb >= 3.4.1 
* matplotlib >= 3.8.2 
* torch >= 2.0.1 
* tqdm >= 4.64.0 
* h5py >= 3.6.0 
* pandas >= 2.1.4 
* seaborn >= 0.13.2 
* ishneholterlib >= 2020.5.29

Our experiments were performed on an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz, with 500 GB RAM, and an
and NVIDIA GeForce RTX 2080 GPU with 24GB of memory.

## Installation Guide
### Setup
To setup the package, which should take several minutes depending on your
internet speed (for downloading dependencies), simply create a new virtual 
environment using:

`conda create --name afsrr python==3.9`


Then activate the virtual environment using:

`conda deactivate`

`conda activate afsrr`

Finally, install the afsrr package using:

`pip install -e .`

This will also install all other dependencies.

Note that specifically for the torch package, you may wish to install it first, 
to make sure that it supports your specific CUDA version in case of using an NVIDIA GPU.
For further instructions please refer to the official installation guide at https://pytorch.org/.

## Demo Guide
As part of this package, and to ensure that everything is working correctly, we provided a demo script that runs a simple experiment on a small subset of the data.
The data is included in this repository and is located under `afsrr\data\demo`.

To run the demo, simply run the demo script located at: `afsrr\afsrr\run\demo\run_demo.py`.

The demo script will train small regression model on the demo dataset, and then validate and test the model.

The training logs of the demo models will be saved in the logs directory of the project, which can be found at: `afsrr\logs\demo`.

The expected output of the demo script should be:
"The demo run has been completed. The afsrr package was successfully installed."

The demo script should take around 5-20 minutes to complete, depending on your specific system.

## Instruction for Use
### Data Generation
All data generation scripts are located at `afsrr\afsrr\run\data_generation`.

You will first need to download the raw databases and generate the processed data to be able to run the experiments.

The default data directory is located at: `afsrr\data`.

The complete data can downloaded from the following sources:

AFDB: https://physionet.org/content/afdb/1.0.0/ -- Please download & extract (if downloaded as a zip) to `afsrr\data\raw\afdb` 

LTAFDB: https://physionet.org/content/ltafdb/1.0.0/ -- Please download & extract (if downloaded as a zip) to `afsrr\data\raw\ltafdb`

NSRDBRR: https://physionet.org/content/nsr2db/1.0.0/ -- Please download & extract (if downloaded as a zip) to `afsrr\data\raw\nsrdbrr`

Alternatively, you can use our script to download the raw data from physionet directly to the appropriate directories.
The script is located at: `afsrr\afsrr\run\data_generation\download_data.py`.

THEW: http://thew-project.org/Database/E-HOL-03-0202-003.html. Note that for the THEW database you
will have to register to the website and agree to the terms of service. Once
registered you will gain access to their datasets. The specific dataset we use is THEW Healthy (E-HOL-03-0202-003).

Please download the THEW dataset to `afsrr\data\raw\thew`

After downloading the raw data, you can generate the processed data using the data generation script
located at: `afsrr\afsrr\run\data_generation\process_data.py`. The processed data will be saved by default at: `afsrr\data\processed`.

Once the processing step is completed we can proceed to run our experiments.

### Running Experiments
The first step should be to train the joint regression model on the training dataset.
This can be done by running the script located at: `afsrr\afsrr\run\experiments\train_joint_regression.py`.

After the completion of the training process we can run in parallel the validation and test experiments.
The validation and test experiments are located at: `afsrr\afsrr\run\experiments\validate_joint_regression.py` and `afsrr\afsrr\run\experiments\test_joint_regression.py` respectively.

Once the validation and test experiments are completed, we can now train a AF screening classifier,
and evaluate how good it is. This can be done by running the script located at: `afsrr\afsrr\run\experiments\af_screening_experiment.py`.

Finally, we can analyze & visualize the results from our various experiments, by running the scripts located at:
`afsrr\afsrr\run\analysis`.

Each experiment will create its own logs directory in the logs directory of the project.
In it, it will automatically log the results for the train/validation/test phases of each experiment.

The defualt logs directory is located at: `afsrr\logs`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.