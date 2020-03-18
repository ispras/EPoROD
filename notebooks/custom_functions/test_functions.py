import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

__all__ = [
    'load_obj',
    'get_train_val_dataframe',
    'visualize_confusion_matrix',
    'remove_flips_from_df',
    'choose_roc_threshold'
]

def choose_roc_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)    
    J = tpr - fpr
    ix = np.argmax(J)
    return thresholds[ix]

def remove_flips_from_df(df):
    flip_list = []
    for i in range(len(df)):
        if "hor" in df.loc[i].filename or "vert" in df.loc[i].filename:
            flip_list.append(False)
        else:
            flip_list.append(True)
    df_no_flips = df[np.array(flip_list)].reset_index().drop("index", axis = 1)
    return df_no_flips

def load_obj(filepath ):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def get_train_val_dataframe(MODEL_DIR, SCENARIO):
    df_dir = os.path.join(MODEL_DIR,'temp_dataframes')
    df_filenames = os.listdir(df_dir)
    df_scenario_filenames = list(filter(lambda x: x.startswith('df_' + SCENARIO), df_filenames))

    df_val_filename = list(filter(lambda x: x.endswith('val.csv'), df_scenario_filenames))[0]
    df_train_filename =list(filter(lambda x: x.endswith('train.csv'), df_scenario_filenames))[0]
    
    df_val = pd.read_csv(os.path.join(df_dir, df_val_filename))
    df_train = pd.read_csv(os.path.join(df_dir, df_train_filename))

    return df_train, df_val


def visualize_confusion_matrix(confusion_matrix, *, img_path = '', img_dpi = 300):
    """
    Visualizes confusion matrix
    
    confusion_matrix: np array of ints, x axis - predicted class, y axis - actual class
                      [i][j] should have the count of samples that were predicted to be class i,
                      but have j in the ground truth
                     
    """
    # Adapted from 
    # https://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    size = confusion_matrix.shape[0]
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    lable_fontzise = 12
    plt.xlabel("predicted", fontsize=lable_fontzise)
    plt.ylabel("ground truth", fontsize=lable_fontzise)

    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.yticks(np.arange(size))
    plt.ylim([- 0.5 + float(size),-0.5])

    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
               plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')
    
    if img_path:
        plt.savefig(img_path, dpi=img_dpi)
