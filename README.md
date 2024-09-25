# Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis
## Overview
This repository contains code and test data in following work:

**"Spike-Tipped Microrobot Enabled Targeted Biopsy for SERS-based Lung Cancer Diagnosis"**  
*(Manuscript currently under review)*

The repository includes a full test set and a demo training set to demonstrate the functionality of the code.

## System Requirements

The code has been tested on the following system configuration:

- **OS**: Ubuntu 22.04.3 LTS
- **Python**: 3.10.12
- **PyTorch**: 2.4.0
- **NumPy**: 1.26.4
- **Matplotlib**: 3.9.0
- **Scikit-learn**: 1.5.0
- **Captum**: 0.7.0 *(used for model attribution analysis)*

## Installation guide

First, install the required dependencies by running the following command:
```bash
pip3 install -r requirements.
```
Then clone this repository:
```
git clone https://github.com/torinonegai/Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis.git
```
Congratulations! After these steps, the code is ready to run.  Installation typically takes **15-30 minutes** depending on your system. 
## Run this code

To quickly test the model with provided test data, simply run the `test.py`:
```
cd Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis
python3 test.py
```
Approximately **3 minutes** are required for collecting and analyzing each spectrum.
The output will include a confusion matrix and the average accuracy of the model, like:
```
[[0.81675393 0.         0.0104712  0.17277487 0.        ]
 [0.         0.99459459 0.         0.00540541 0.        ]
 [0.01327434 0.01769912 0.96017699 0.00884956 0.        ]
 [0.         0.         0.00625    0.9875     0.00625   ]
 [0.00497512 0.         0.         0.03482587 0.960199  ]]

Average accuracy: 0.9428868120456906
```
For training the model and conducting attribution analysis, run the training process with `main.py` and perform attribution analysis with `attibution.py`.

To train the model  with the demo training set:
```bash
python3 main.py
```
The output will include a model saved as `tmp.pth` along with test loss and accuracy, like:
```
Test loss: 0.2718150317668915 
Test accuracy: 0.8805815160955348
```
Note: To use your own dataset, modify lines 9 and 10 in `main.py` to point to the appropriate dataset filenames.
## License

This project is covered under the **Apache 2.0 License**.