# Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis
## Instructions
This repository contains code and test data in following work:

"Spike-Tipped Microrobot Enabled Targeted Biopsy for SERS-based Lung Cancer Diagnosis",  (In review)

All of the test set and a demo train set is uploaded.
## System Requirements
The code has been tested on Linux operating systems, with the following versions of software dependencies:
- **OS**: Ubuntu 22.04.3 LTS
- **Python**: 3.10.12
- **PyTorch**: 2.4.0
- **NumPy**: 1.26.4
- **Matplotlib**: 3.9.0
- **Scikit-learn**: 1.5.0
- **Captum**: 0.7.0 *(used for model attribution analysis)*
## Installation guide
First install the dependencies software:
```
pip3 install -r requirements.
```
Then clone the repository:
```
git clone
```
Congratulations! Now the code is ready for running.
Typical install time is about 15-30min
## Run this code
* To quickly test the model with provided test data, simply run the `test.py`.
* For training the model and conducting attribution analysis, 
run the training process with `main.py` and perform attribution analysis with `attibution.py`