<b>Description</b>

This repository contains notebooks for CNN training and evaluating for retina differentiation project.
All neccecary additional python libraries are listed in requirements txt. 

<b>Directories</b>

datasets -- contains working datasets. <br/>
notebooks -- contains notebooks for CNNs training and testing<br/>
models -- contains the models, their metric scores and train histories<br/>

<b>Dataset</b>

Current working dataset is in the diretory datasets/dataset_mixed. It has 5 subfolders:<br/>

train -- dataframe and images for trainig<br/>
train_flipped -- the same as "train", but the images of "non-retina" class got vertivally and horisontally flipped duplicates<br/>
test -- dataframe and images for testing<br/>
crossval_10 -- dataframes for 10-flod cross-validation. Note, that images for this procedure are taken from "train_flipped" dir<br/>
not_used -- images, which were sorted out during preprocessing (mostly non-containg anything).<br/>

This repository doesn't contain images -- to train the CNNs, they should be added manually in train and test directories.<br/>


<b>Usage</b>

For CNN training run the file "notebooks/training/train.ipynb" in jupyter-lab. The model specifications are listed in the file "notebooks/training/train-config.json". The fields from this config will be referred in this guide as config["config_field"].<br/>

For CNN testing run the file "notebooks/testing/test.ipynb"<br/>

Notebook "notebooks/training/train.ipynb" creates a directory named as config["model_name"] with in config["all_models_dir"] (by default it's "models"). This directory contains trained models (*.h5 files) (one with best score on train dataset and one obtained after the last epoch) , wodel_weights (for each epoch), history files with train scores for each epoch. After running "notebooks/testing/test.ipynb" test scores and images of the learning curves are put there.<br/>
