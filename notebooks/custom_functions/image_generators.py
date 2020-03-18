from keras.preprocessing.image import ImageDataGenerator
from train_functions import *

def get_generator(config, dataframe, shuffle = False, *, augmentation = False):
    
    samplewise_std_normalization, samplewise_center,\
    featurewise_std_normalization, featurewise_center,\
    rescale = normalization_mode_setter(config["normalization_mode"])
          
    if augmentation:
        horizontal_flip = True
        vertical_flip=True
        rotation_range=5
        width_shift_range=0.1
        height_shift_range=0.1
        zoom_range=0.1
    else:
        horizontal_flip=False
        vertical_flip=False
        rotation_range=0.0
        width_shift_range=0.0
        height_shift_range=0.0
        zoom_range=0.0
    
    datagen = ImageDataGenerator(
        samplewise_std_normalization = samplewise_std_normalization, 
        samplewise_center = samplewise_center,
        featurewise_std_normalization = featurewise_std_normalization,
        featurewise_center = featurewise_center,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale = rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        fill_mode='nearest')

    generator = datagen.flow_from_dataframe(
        dataframe = dataframe,
        x_col="filename",
        y_col="output_classes",
        directory = config["dataset_dir"],
        target_size=(config["image_size"], config["image_size"]),
        batch_size=config["batch_size"],
        class_mode=config["class_mode"],
        #color_mode="grayscale"
        shuffle=shuffle
    )
    
    return generator
    
    
