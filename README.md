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
git clone https://github.com/torinonegai/Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis.git
```
Congratulations! Now the code is ready for running. The Typical install time is about **15-30 minutes**.
## Run this code
* To quickly test the model with provided test data, simply run the `test.py`.
```
cd Spike-Tipped-Microrobot-Enabled-Targeted-Biopsy-for-SERS-based-Lung-Cancer-Diagnosis
python3 test.py
```
Typically, the time for collecting and analyzing a spectra is about **3 minutes**.
The output is the confusion matrix alone with the average accuracy like 
```
[[0.81675393 0.         0.0104712  0.17277487 0.        ]
 [0.         0.99459459 0.         0.00540541 0.        ]
 [0.01327434 0.01769912 0.96017699 0.00884956 0.        ]
 [0.         0.         0.00625    0.9875     0.00625   ]
 [0.00497512 0.         0.         0.03482587 0.960199  ]]
 
Average accuracy: 0.9428868120456906
```
* For training the model and conducting attribution analysis, run the training process with `main.py` and perform attribution analysis with `attibution.py`