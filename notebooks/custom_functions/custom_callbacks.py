import pickle
import os
import sys
import json
import numpy as np

from keras.callbacks import Callback
from keras.callbacks import LambdaCallback
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


def save_obj(obj, filepath ):
    with open(filepath, 'w+') as file:
        json.dump(obj, file)

def load_obj(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def save_pkl(obj, filepath ):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(filepath ):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class EveryEpochSaver(Callback):
    
    def __init__(self, filename_template, target_dir):
        self.filename_template = filename_template
        self.target_dir = target_dir
        
    def on_epoch_end(self, epoch, logs={}):
            filename = "_".join([self.filename_template, "epoch", str(epoch), ".hdf5"])
            file_path = os.path.join(self.target_dir, filename)
            self.model.save_weights(file_path)
    
class modelTracker(Callback):
    def __init__(self,
                 history_dir,
                 train_gen,
                 val_gen,
                 model_name,
                 model_dir,
                 *,
                 filename ='history.json', 
                 monitor = "val_loss",
                ):
        
        self.history_dir = history_dir
        self.filename = filename
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.histories = {}

        self.histories['train_labels'] = train_gen.labels
        self.histories['val_labels'] = val_gen.labels
        self.histories['train_loss'] = []
        self.histories['val_loss'] = []
        self.histories['train_ROCAUC'] = []
        self.histories['val_ROCAUC'] = []
        self.histories['train_probs'] = []
        self.histories['val_probs'] = []
        
        self.model_name = model_name
        self.model_dir = model_dir
        self.monitor = monitor
        self.best_score = None
        
        self.best_model_filepath = os.path.join(model_dir, "_".join(["best", model_name,".h5"]))
    
    def save_best_model(self, epoch):
        
        if self.monitor == "train_loss" or self.monitor == "val_loss":
            quality_to_compare = - self.histories[self.monitor][epoch]
        else:
            quality_to_compare = self.histories[self.monitor][epoch]
            
        if epoch == 0:
            self.best_score = quality_to_compare
        else:       
            if quality_to_compare > self.best_score:
                self.best_score = quality_to_compare
                self.model.save(self.best_model_filepath)
                print("Model with best %s is saved"%(self.monitor))
        return
        
    def on_epoch_end(self, epoch, logs={}):
        
        train_probs = self.model.predict_generator(self.train_gen).astype(np.float64, copy=False)
        val_probs = self.model.predict_generator(self.val_gen).astype(np.float64, copy=False) 
        self.histories['train_probs'].append(train_probs.flatten().tolist())
        self.histories['val_probs'].append(val_probs.flatten().tolist())
        
        train_rocauc = roc_auc_score(self.train_gen.labels, train_probs)
        val_rocauc = roc_auc_score(self.val_gen.labels, val_probs)
        self.histories['train_ROCAUC'].append(train_rocauc)
        self.histories['val_ROCAUC'].append(val_rocauc)
        
        train_loss = log_loss(self.train_gen.labels, train_probs)
        val_loss = log_loss(self.val_gen.labels, val_probs)
        self.histories['train_loss'].append(train_loss)
        self.histories['val_loss'].append(val_loss)

        save_obj(self.histories, os.path.join(self.history_dir, self.filename))
        
        self.save_best_model(epoch)
        
        print("Train--Val loss: %f -- %f, Train--Val ROCAUC: %f -- %f"% (train_loss, val_loss, train_rocauc, val_rocauc))
        
        return

        
class saveModelCallback:
    def __init__(self, *, model_name, model_dir, model, metrics = 'val_accurcy',   tolerance= 0.05):
        self.model_path = os.path.join(model_dir, model_name)
        self.best_val_acc = 0
        self.best_val_loss = [sys.float_info.max]
        self.tolerance = tolerance
        self.model = model
        self.callback = LambdaCallback(on_epoch_end=self.saveModel)
        
    def saveModel(self, epoch, logs):
        val_acc = logs['val_accuracy']
        acc = logs['accuracy']
        if (abs(val_acc - acc) < self.tolerance) and (val_acc > self.best_val_acc):
            self.best_val_acc = val_acc
            print('Saving model with acc=', acc, 'and val_acc=', val_acc)         
            self.model.save(self.model_path)