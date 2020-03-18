import os
import shutil
import pandas as pd
import numpy as np
import glob
import pickle
import datetime
import socket
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications import ResNet50
from keras.applications import VGG19, VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import BatchNormalization, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop, SGD

__all__ = [
    'get_class_weigts',
    'normalization_mode_setter',
    'create_temp_dfs',
    'get_optimizer',
    'make_model_config',
    'make_place_for_the_model',
    'create_kfold_dataframes',
    'train_val_split',
    'import_kfold_dataframes',
    "make_model"
]

def get_class_weigts(generator):
    
    #returns the weights for each class, in
    number_of_classes = len(generator.class_indices.keys())
    
    retina_index = generator.class_indices['retina']
    non_retina_index = generator.class_indices['non-retina']
       
    number_of_retinas = np.sum(np.array(generator.labels) == retina_index)
    number_of_non_retinas = np.sum(np.array(generator.labels) == non_retina_index)
    
    if number_of_classes == 2:
         class_weights = {   
                         retina_index: 1.,
                         non_retina_index: number_of_retinas / number_of_non_retinas                
        }
    elif number_of_classes == 3:
        
        satisfactory_index = generator.class_indices['Satisfactory']
        number_of_satisfactory = np.sum(np.array(generator.labels) == satisfactory_index)
        
        class_weights = {
            retina_index: 1.,
            non_retina_index: number_of_retinas / number_of_non_retinas,
            satisfactory_index: number_of_retinas / number_of_satisfactory
        }
    else:
        raise Exception("Wrong number of classes!")

    return class_weights

def normalization_mode_setter(NORMALIZATION_MODE):
    if NORMALIZATION_MODE == 'samplewise':
        samplewise_std_normalization = True
        samplewise_center = True
    
        featurewise_std_normalization = False
        featurewise_center = False

        rescale = None
    
    elif NORMALIZATION_MODE == 'featurewise':
        samplewise_std_normalization = False
        samplewise_center = False
    
        featurewise_std_normalization = True
        featurewise_center = True

        rescale = None
    
    elif NORMALIZATION_MODE == 'rescale':
        samplewise_std_normalization = False
        samplewise_center = False
    
        featurewise_std_normalization = False
        featurewise_center = False

        rescale = 1./255.
        
    return samplewise_std_normalization, samplewise_center, featurewise_std_normalization, featurewise_center, rescale


def create_temp_dfs(df_path, temp_dataframes_dir, model_name, train_val_ratio):
 
    if os.path.exists(temp_dataframes_dir):
        shutil.rmtree(temp_dataframes_dir)
    os.mkdir(temp_dataframes_dir)
    df = pd.read_csv(df_path)
    X = df.index.values
    y = df.output_classes.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_ratio)
    df_train_path = os.path.join(temp_dataframes_dir, "_".join(["df", model_name, "train.csv"]))
    df_val_path = os.path.join(temp_dataframes_dir, "_".join(["df", model_name, "val.csv"]))
    df_train = df.loc[X_train]
    df_val = df.loc[X_test]
    df_train.to_csv(df_train_path, index = False)
    df_val.to_csv(df_val_path, index = False)    
                                    
    return df_train, df_val

def train_val_split(df, save_dir, config_dict):
    train_val_ratio = config_dict["train_val_ratio"]
    model_name = config_dict["model_name"]
    X = df.index.values
    y = df.output_classes.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_ratio)
    df_train_path = os.path.join(save_dir, "_".join(["df", model_name, "train.csv"]))
    df_val_path = os.path.join(save_dir, "_".join(["df", model_name, "val.csv"]))
    df_train = df.loc[X_train]
    df_val = df.loc[X_test]
    df_train.to_csv(df_train_path, index = False)
    df_val.to_csv(df_val_path, index = False)
    
    return df_train, df_val

def import_kfold_dataframes(config_dict):
    
    dataframes_dir = config_dict["cross_validation"]["dataframes_dir"]
    split_dirs = os.listdir(dataframes_dir)
    n_splits = config_dict["cross_validation"]["folds"]
        
    dfs_train=[]
    dfs_val=[]   
    for i in range(n_splits):
        df_dir = os.path.join(dataframes_dir, "split_" + str(i))
        if not os.path.exists(df_dir):
            raise Exception("Some split dirs are missing") 
        dfs_train.append(pd.read_csv(os.path.join(df_dir,  "_".join(["df","train", str(i),".csv"]))))
        dfs_val.append(pd.read_csv(os.path.join(df_dir,  "_".join(["df","val", str(i),".csv"]))))
        
    return dfs_train, dfs_val

def create_kfold_dataframes(config_dict):
    model_name = config_dict["model_name"]
    model_dir = os.path.join(config_dict["all_models_dir"], model_name)
    df_path = config_dict["dataframe_path"]
    n_splits = config_dict["cross_validation"]["folds"]
    
    df = pd.read_csv(df_path)
    X = df.index.values
    y = df.output_classes.values
    kf = KFold(n_splits = n_splits, shuffle = True)
    kf.get_n_splits(X)
    
    split_number = 0
    dfs_train = []
    dfs_test = []
    for train_index, test_index in kf.split(X):
        split_path = os.path.join(model_dir, "_".join(["split", str(split_number)]))
        temp_dataframes_dir = os.path.join(split_path, "temp_dataframes")
        if not os.path.exists(temp_dataframes_dir):
             raise Exception("Some split dirs are missing!")
        df_train_path = os.path.join(temp_dataframes_dir,"_".join(["df", model_name, "train.csv"]))
        df_test_path = os.path.join(temp_dataframes_dir, "_".join(["df", model_name, "val.csv"]))
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]
        df_train.to_csv(df_train_path, index = False)
        df_test.to_csv(df_test_path, index = False)
        dfs_train.append(df_train)
        dfs_test.append(df_test)
        
        split_number += 1
        
    print(len(dfs_train[0])) 
    return dfs_train, dfs_test
               
def get_optimizer(opt_dict):
    if opt_dict['optimizer_type'] == 'Adam':
        return Adam(learning_rate=opt_dict['learning_rate'], 
                    beta_1=opt_dict['beta_1'], 
                    beta_2=opt_dict['beta_2'], 
                    amsgrad=opt_dict['amsgrad']
                    )
    elif opt_dict['optimizer_type'] == 'SGD':
        return SGD(learning_rate=opt_dict['learning_rate'], 
                   momentum=opt_dict['momentum'], 
                   nesterov=opt_dict['nesterov'] 
                        ) 
    else:
        raise Exception("Unknown type of optimizer: ", opt_dict['optimizer_type']) 

def make_model_config(train_config):
    
    NOW = datetime.datetime.now()
    config_dict = train_config.copy()
    config_dict["computer_name"] = socket.gethostname()
    config_dict["creation_time"] =  NOW.strftime("%T")
    config_dict["creation_date"] =  NOW.strftime("%d %B %Y")
    config_dict["training_is_finished"] = False
    return config_dict
    

def make_place_for_the_model(config_dict, MODEL_OVERWRITE_MODE):

    MODEL_NAME = config_dict['model_name']
    ALL_MODELS_DIR = config_dict['all_models_dir']
    creation_time = config_dict['creation_time']
    creation_date = config_dict['creation_date']
    computer_name = config_dict['computer_name']

    #If model name is empty generate it from computer name, date and time
    if MODEL_NAME == '':
        MODEL_NAME = computer_name + '_' + creation_time + '_' + creation_date
    
    model_dir = os.path.join(ALL_MODELS_DIR, MODEL_NAME)
   
    #Creation model dir
    if MODEL_OVERWRITE_MODE not in ['on', 'off']:
        raise Exception('Wrong MODEL_OVERWRITE_MODE!')
    elif MODEL_OVERWRITE_MODE == 'off' and os.path.exists(model_dir):
         raise Exception('Model\'s dir exists!')            
    else:
        if os.path.exists(model_dir):
            pass
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        
    if config_dict["cross_validation"]["used"]:
        model_dir_list = [os.path.join(model_dir, "_".join(["split", str(k)]))\
                              for k in range(config_dict["cross_validation"]["folds"])]
        for target_dir in model_dir_list:
            os.mkdir(target_dir)
            pass
    else:
         model_dir_list=[model_dir]    
    
    history_dirs = []
    temp_dataframes_dirs = []
    model_weights_dirs = []
    for target_dir in model_dir_list:
        temp_dataframes_dir = os.path.join(target_dir, 'temp_dataframes')
        history_dir = os.path.join(target_dir, 'train_history')
        model_weights_dir = os.path.join(target_dir, 'model_weights')
        os.mkdir(temp_dataframes_dir)
        os.mkdir(history_dir)
        os.mkdir(model_weights_dir)
        history_dirs.append(history_dir)
        temp_dataframes_dirs.append(temp_dataframes_dir)
        model_weights_dirs.append(model_weights_dir)
        if config_dict["cross_validation"]["used"]:
            split_name = os.path.split(target_dir)[-1]
            temp_df_dir = os.path.join(config_dict["cross_validation"]["dataframes_dir"], split_name)
            if not os.path.exists(temp_df_dir):
                raise Exception("Some split dirs are missing!")
            for file_name in os.listdir(temp_df_dir):
                file_path = os.path.join(temp_df_dir, file_name)
                dest_dir = os.path.join(target_dir, 'temp_dataframes')
                shutil.copy(file_path, dest_dir)
                                                         
    with open(os.path.join(model_dir,"_".join([MODEL_NAME,"config.json"])), "w+") as write_file:
        json.dump(config_dict, write_file)

    return model_dir, temp_dataframes_dirs, history_dirs, model_weights_dirs

        
def make_model(config, pretrainedModelType):
    #Importing pretrained model. If pretrained_model_path is empty,
    # the one from Keras library is used, with several layers being appended
    if config["pretrained_model_path"]: 
        model = load_model(config["pretrained_model_path"])
        print('Got pretrained model from %s'% pretrained_model_path)
    else:
        pretrained_model = pretrainedModelType(
                                                weights='imagenet', 
                                                include_top=False
                                                )        
        x = pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(config["last_layer_dropout_rate"])(x)
        x = Dense(config["last_layer_output_dim"])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(config["last_layer_dropout_rate"])(x)

        output_channels_number = 1
        predictions = Dense(output_channels_number, activation=config["activation"])(x)

        model = Model(inputs=pretrained_model.input, outputs=predictions)
    
    #Freezeing some layers
    trainable_layers = [model.layers[i] \
                        for i in range(config["trainable_layers"]["from"],config["trainable_layers"]["to"])]

    non_trainable_layers = list(set(model.layers) - set(trainable_layers))

    for layer in trainable_layers:
        layer.trainable = True

    for layer in non_trainable_layers:
        layer.trainable = False
    
    #Defining the optimizer
    optimizer = get_optimizer(config['opt_dict'])
    model.compile(loss = config['loss'],
              optimizer = optimizer,
              metrics=['accuracy'])
    print(optimizer)
        
    return model