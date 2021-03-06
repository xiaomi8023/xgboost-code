# Classification of Earthquakes, Explosions and Mine Collapses Based on XGBoost Algorithm

by Tingting Wang<sup>1</sup>, Yinju Bian<sup>1</sup>, Yixiao Zhang <sup>1</sup>, Xiaolin Hou<sup>1</sup>

<sup>1</sup>Institute of Geophysics, China Earthquake Administration, Beijing 100081, China.

Corresponding author affiliation and e-mail:

Yinju Bian

Institute of Geophysics, China Earthquake Administration.
No.5 South Minzu Unversity Road, Haidian District, Beijing 100081, China

bianyinju@cea-igp.ac.cn

This repository contains the source code to perform prediction and evaluation with example data. 

## Content
- file folder: source
- Classify_svm.py  
Python script containing the SVM function to predict earthquake events classes.
- Classify_xgboost.py  
Python script containing the XGBoost function to predict earthquake events classes.
- file folder: data 
- example36.csv  
This file contains 36 feature extraction training sets for different types of seismic events;
- example201.csv  
This file contains 201 waveform spectrum training sets for different types of seismic events;
- svmmodel_36.pkl

   SVM Pre-trained model to predict seismic event based on 36 features trainning set.

- svmmodel_201.pkl
 
   SVM Pre-trained model to predict seismic event based on 201 features trainning set.
- xgbmodel_36.pkl
 
   xgboost Pre-trained model to predict seismic event based on 36 features trainning set.

- xgbmodel_201.pkl

   xgboost Pre-trained model to predict seismic event based on 201 features trainning set.
   
## install  
The code has been tested using packages of:  
- Python (version 3.7)
- numpy (1.19.2)
- pandas (1.0.1)
- scipy (1.6.1)
- scikit-learn (0.24.0)
- matplotlib (3.1.3)
- joblib (0.15.1)
- xgboost(1.3.1)

## How to run the code? Training and Evaluation

Running the code `Classify_svm.py` will perform the prediction and evaluation. to predict the results of data. 

Running the code `Classify_xgboost.py` will perform the prediction and evaluation. to predict the results of data. 


## License

The following legal note is restricted solely to the content of the named files. It cannot
overrule licenses from the Python standard distribution modules, which are imported and
used therein.

BSD 3-clause license

Copyright (c) 2021 Tingting Wang, Yinju Bian, Yixiao Zhang , and Xiaolin Hou.
All rights reserved.


