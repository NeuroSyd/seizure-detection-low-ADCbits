# seizure-detection-low-ADCbits
Epileptic Seizure Detection on Low-Precision Electroencephalogram Signals

##### Code to reproduce results reported in our paper published as:
* Truong, N. D. and O. Kavehei (2019). Epileptic Seizure Detection on Low-Precision Electroencephalogram Signals. Under review with *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*.

* Truong, N. D. and O. Kavehei (2019). Low Precision Electroencephalogram for Seizure Detection with Convolutional Neural Network. *IEEE International Conference on Artificial Intelligence Circuits and Systems*.}

#### Requirements

* numpy==1.11.0
* stft==0.5.2
* mne==0.11.0
* pandas==0.24.2
* Keras==2.1.6
* hickle==3.4.3
* scipy==1.0.1
* scikit_learn==0.21.3

#### How to run the code
1. Set the paths in \*.json files. Copy files in folder "copy-to-CHBMIT" to your CHBMIT dataset folder.

2. Leave-one-seizure-out cross-validation or test.
> python3 main.py --mode [cv,test] --dataset DATASET --adcbits [number-of-ADCbits]
* DATASET can be FB, CHBMIT.
