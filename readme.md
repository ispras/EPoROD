<b>Description</b>

This repository contains notebooks for CNN training and evaluating for retina differentiation project.
All neccecary additional python libraries are listed in requirements txt. 

<b>Directories</b>

datasets -- contains working datasets<br/>
notebooks -- contains notebooks for CNNs training and testing<br/>
models -- contains the models, their metric scores and train histories<br/>

<b>Usage</b>

For CNN training run the file notebooks/training/train.ipynb in jupyter-lab. The model specifications are listed in file train-config.json<br/>
For CNN cross-validation run the file notebooks/training/cross_val_train.ipynb in jupyter-lab. The model specifications are listed in file cross_val_train_config.json<br/>
For CNN testing run the file notebooks/testing/test.ipynb<br/>
For computing CNN lectures after cross validation used the files crossval_eval.ipynb and McNemar_test.ipynb in jupyter-lab in /notebooks.testing directory<br/>
The images for training and testing as well as the dataframes with filenames and labels should be stored in the directory /dataset/dataset_mixed/train and /dataset/dataset_mixed/test<br/>
