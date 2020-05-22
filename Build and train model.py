#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from glob import glob
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.initializers import GlorotNormal
from itertools import chain
from math import ceil
import cv2 as cv
from random import uniform, randint

batch_size = 16  # Used for training
validation_batch_size = 64
initial_epoch = 0  # Used to resume training from a saved set of weights, if available
n_epochs = 10  # Max number of epochs for training during the current run (i.e. counting after initial_epoch)
augmentation_rate = 0  # Number of synthetic images that will be produced for every image in the training dataset
dataset_root = '/media/fanta/52A80B61A80B42C9/Users/fanta/datasets'
augmentation_dir = dataset_root + '/data/augmented'  # TODO make it portable
augmentation_batch_size = 229  # A divisor of 2290, the number of positive images in the dataset
weight_path = 'weights/weights.{epoch:04d}-{val_loss:.2f}.hdf5'
pd.set_option('display.max_rows', 100)

# Check there is a GPU available
n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
assert n_gpus >= 1

# Read the dataset metadata
all_xray_df = pd.read_csv(dataset_root + '/data/Data_Entry_2017.csv')
all_xray_df = all_xray_df.drop(columns=all_xray_df.columns[-1])
# all_xray_df.sample(10)

# Add a 'path' column to the metadata dataframe holding the full path to dataset images
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join(dataset_root + '/data', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
# l_xray_df.sample(3)


# One-hot encode the findings for each row in the metadata dataframe, adding more columns as needed
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for c_label in all_labels:
    all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)
# all_xray_df.sample(10)

def split_dataset(dataset, test_size):
    patient_id = 'Patient ID'
    pneumonia = 'Pneumonia'
    # For each patient, count how many positive samples in the dataset, and shuffle the resulting serie
    patient_positive_count = shuffle(all_xray_df.groupby(patient_id)[pneumonia].agg('sum'))
    patient_positive_count.reset_index(inplace=True, drop=True)
    n_positive_samples = sum(dataset[pneumonia])
    n_wanted = int(n_positive_samples * test_size)
    selected_patients = set()
    n_selected = 0
    for id, count in patient_positive_count.items():
        selected_patients.add(id)
        n_selected += count
        if n_selected > n_wanted:
            break
    test_set = shuffle(dataset[dataset[patient_id].isin(selected_patients)])
    test_set.reset_index(inplace=True, drop=True)
    training_set = shuffle(dataset[~dataset[patient_id].isin(selected_patients)])
    training_set.reset_index(inplace=True, drop=True)
    assert len(training_set) + len(test_set) == len(dataset)
    return training_set, test_set


train_df, test_df = split_dataset(all_xray_df, .2)


def enforce_classes_ratio(dataset_df, ratio):
    # Reduce the training set removing enough negative cases to remain with a training set with 50% positive and 50% negative
    count_train_pos = sum(dataset_df.Pneumonia)

    res_df = dataset_df[dataset_df.Pneumonia == 0][:int(count_train_pos * ratio)].append(
        dataset_df[dataset_df.Pneumonia == 1])
    res_df = shuffle(res_df)
    res_df.reset_index(inplace=True, drop=True)
    return res_df


train_df = enforce_classes_ratio(train_df, 4)
test_df = enforce_classes_ratio(test_df, 4)

train_pos = sum(train_df.Pneumonia)
train_neg = len(train_df.Pneumonia) - train_pos
class_weight = {1: float(train_neg) / (train_pos + train_neg), 0: float(train_pos) / (train_pos + train_neg)}


def perturbate_and_preprocess(image, data_format=None):
    # TODO try also improving the image contrast/dynamic range; in case, it must be done in the validation dataset too
    flip = True
    stretch = .15  # Both vertical and horizontal
    shear = .05  # Horizontal only
    rotate = 7  # In degrees, positive is clockwise
    translate_x = .05  # Fraction of the image width
    translate_y = .0  # Fraction of the image height

    rows, cols = image.shape[0], image.shape[1]
    # Flip
    if flip and randint(0, 1) == 1:
        image = cv.flip(image, 1)
    # Stretch
    s_x = uniform(1 - stretch, 1 + stretch)
    s_y = uniform(1 - stretch, 1 + stretch)
    stretching = np.array([[s_x, 0, 0], [0, s_y, 0]])
    image = cv.warpAffine(image, stretching, (rows, cols), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # Shear
    shift = cols * uniform(-shear, shear)
    pts1 = np.float32([(0, 0), (0, cols - 1), (rows - 1, 0)])
    pts2 = np.float32([(-shift, 0), (shift, cols - 1), (rows - 1 - shift, 0)])
    shearing = cv.getAffineTransform(pts1, pts2)
    image = cv.warpAffine(image, shearing, (rows, cols), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # Rotate
    angle = uniform(-rotate, rotate)
    rotation = cv.getRotationMatrix2D(center=((cols - 1) / 2., (rows - 1) / 2.), angle=angle, scale=1.)
    image = cv.warpAffine(image, rotation, (rows, cols), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # Translate
    t_x = uniform(-cols * translate_x, cols * translate_x) + cols * (1 - s_x) / 2
    t_y = uniform(-rows * translate_y, rows * translate_y) + rows * (1 - s_y) / 2
    translation = np.array([[1, 0, t_x], [0, 1, t_y]])
    image = cv.warpAffine(image, translation, (rows, cols), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # VGG16 pre-processing
    image = preprocess_input(image, data_format)
    return image


def make_train_gen(train_df, dataset_root, batch_size):
    idg = ImageDataGenerator(preprocessing_function=perturbate_and_preprocess)
    train_gen = idg.flow_from_dataframe(dataframe=train_df,
                                        directory=dataset_root,
                                        x_col='path',
                                        y_col='Pneumonia',
                                        class_mode='raw',  # TODO should I use binary instead?
                                        target_size=(224, 224),  # Input size for VGG16
                                        batch_size=batch_size,
                                        shuffle=True)
    return train_gen


def make_val_gen(val_df, dataset_root, batch_size):
    idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_gen = idg.flow_from_dataframe(dataframe=val_df,
                                      directory=dataset_root,
                                      x_col='path',
                                      y_col='Pneumonia',
                                      class_mode='raw',  # TODO should I use binary instead?
                                      target_size=(224, 224),  # Input size for VGG16
                                      batch_size=batch_size,
                                      shuffle=True
                                      )
    return val_gen


train_gen = make_train_gen(train_df, dataset_root, batch_size)
val_gen = make_val_gen(test_df, dataset_root, validation_batch_size)

## May want to look at some examples of our augmented training data.
## This is helpful for understanding the extent to which data is being manipulated prior to training,
## and can be compared with how the raw data look prior to augmentation

"""t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:, :, 0], cmap='bone')
    if c_y == 1:
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('No Pneumonia')
    c_ax.axis('off')

plt.show()"""

# Build the model

base_model = VGG16(include_top=True, weights='imagenet')
base_model.summary()

transfer_layer = base_model.get_layer('block5_pool')
vgg_model = Model(inputs=base_model.input,
                  outputs=transfer_layer.output)

for layer in vgg_model.layers[0:14]:
    layer.trainable = False

for layer in vgg_model.layers:
    print(layer.name, layer.trainable)

retrofitted_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
retrofitted_model.add(vgg_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
retrofitted_model.add(Flatten())

retrofitted_model.add(Dropout(0.333))
retrofitted_model.add(
    Dense(1024, activation='relu', kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal()))
retrofitted_model.add(Dropout(0.333))
retrofitted_model.add(Dense(512, activation='relu', kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal()))
retrofitted_model.add(Dropout(0.333))
retrofitted_model.add(Dense(256, activation='relu', kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal()))
retrofitted_model.add(
    Dense(1, activation='sigmoid', kernel_initializer=GlorotNormal(), bias_initializer=GlorotNormal()))

retrofitted_model.summary()

all_weights_paths = {os.path.basename(x): x for x in
                     glob(os.path.join('weights', 'weights*.hdf5'))}

if len(all_weights_paths) > 0:
    latest_and_greatest = max(all_weights_paths.keys())
    retrofitted_model.load_weights(all_weights_paths[latest_and_greatest])
    initial_epoch = int(latest_and_greatest[8:12])
    print('Resuming with epoch', initial_epoch + 1, 'from weights previously saved in', latest_and_greatest)

optimizer = Adam(lr=.5e-4)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

retrofitted_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

checkpoint = ModelCheckpoint(weight_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True)

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=10)

callbacks_list = [checkpoint]

steps_per_epoch = ceil(len(train_df) * (1. + augmentation_rate) / batch_size)
validation_steps = ceil(float(len(test_df)) / validation_batch_size)

val_pos = sum(test_df.Pneumonia)
val_neg = len(test_df) - val_pos

print()
print('Training Set ---------------------------------')
print('Positive samples {}'.format(train_pos))
print('Negative samples {}'.format(train_neg))
print('Total samples {}'.format(len(train_df)))
print('Total samples after augmentation {}'.format(steps_per_epoch * batch_size))
print('Augmentation rate {}'.format(augmentation_rate))
print('Validation Set -------------------------------')
print('Positive samples {}'.format(val_pos))
print('Negative samples {}'.format(val_neg))
print('Total samples in batches {}'.format(validation_steps * validation_batch_size))
print('----------------------------------------------')
print()

history = retrofitted_model.fit(x=train_gen,
                                class_weight=class_weight,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=validation_steps,
                                initial_epoch=initial_epoch,
                                epochs=initial_epoch + n_epochs,
                                callbacks=callbacks_list)


# Define a function here that will plot loss, val_loss, binary_accuracy, and val_binary_accuracy over all of
# your epochs:
def plot_history(history):
    epochs = [i + 1 for i in history.epoch]
    train_precision = np.array(history.history['precision'])
    train_recall = np.array(history.history['recall'])
    val_precision = np.array(history.history['val_precision'])
    val_recall = np.array(history.history['val_recall'])
    train_F1 = 2. * (train_precision * train_recall) / (
            train_precision + train_recall)
    val_F1 = 2. * (val_precision * val_recall) / (val_precision + val_recall)

    _, ax1 = plt.subplots()
    ax1.set_title('Training and Validation Metrics')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)
    ax1.plot(epochs, history.history["loss"], label="Train. loss", color='greenyellow')
    ax1.plot(epochs, history.history["val_loss"], label="Val. loss", color='darkolivegreen')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1')
    ax2.set_xticks(epochs)
    ax2.plot(epochs, train_F1, label="Train. F1", color='magenta')
    ax2.plot(epochs, val_F1, label="Val. F1", color='darkmagenta')

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')


plot_history(history)
plt.show()


def load_pretrained_model(vargs):
    # model = VGG16(include_top=True, weights='imagenet')
    # transfer_layer = model.get_layer(lay_of_interest)
    # vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

    # Todo

    return vgg_model


# In[ ]:


def build_my_model(vargs):
    # my_model = Sequential()
    # ....add your pre-trained model, and then whatever additional layers you think you might
    # want for fine-tuning (Flatteen, Dense, Dropout, etc.)

    # if you want to compile your model within this function, consider which layers of your pre-trained model, 
    # you want to freeze before you compile 

    # also make sure you set your optimizer, loss function, and metrics to monitor

    # Todo

    return None


## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
## to output class activation maps to aid in clinical interpretation of your model's results

# ##### After training for some time, look at the performance of your model by plotting some performance statistics:
# 
# Note, these figures will come in handy for your FDA documentation later in the project

# In[ ]:


## After training, make some predictions to assess your model's overall performance
## Note that detecting pneumonia is hard even for trained expert radiologists, 
## so there is no need to make the model perfect.
# my_model.load_weights(weight_path)
# pred_Y = new_model.predict(valX, batch_size=32, verbose=True)


# In[ ]:


def plot_auc(t_y, p_y):
    ## Hint: can use scikit-learn's built in functions here like roc_curve

    # Todo

    return


## what other performance statistics do you want to include here besides AUC?


# def ... 
# Todo

# def ...
# Todo


# In[ ]:


## plot figures

# Todo


# Once you feel you are done training, you'll need to decide the proper classification threshold that optimizes your model's performance for a given metric (e.g. accuracy, F1, precision, etc.  You decide) 

# In[ ]:


## Find the threshold that optimize your model's performance,
## and use that threshold to make binary classification. Make sure you take all your metrics into consideration.

# Todo


# In[ ]:


## Let's look at some examples of predicted v. true with our best model: 

# Todo

# fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))
# i = 0
# for (c_x, c_y, c_ax) in zip(valX[0:100], testY[0:100], m_axs.flatten()):
#     c_ax.imshow(c_x[:,:,0], cmap = 'bone')
#     if c_y == 1: 
#         if pred_Y[i] > YOUR_THRESHOLD:
#             c_ax.set_title('1, 1')
#         else:
#             c_ax.set_title('1, 0')
#     else:
#         if pred_Y[i] > YOUR_THRESHOLD: 
#             c_ax.set_title('0, 1')
#         else:
#             c_ax.set_title('0, 0')
#     c_ax.axis('off')
#     i=i+1


# In[ ]:


## Just save model architecture to a .json:

model_json = retrofitted_model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)

# Screenshot from 2020-05-22 21-07-46 1-4 pos-neg, only 1 layer, no augment
# Screenshot from 2020-05-22 21-50-40 Added 3 layers with drop-off and image perturbation (no real augment)
# Screenshot from 2020-05-22 22-14-27 Added fine tuning of the last 3 layers of VGG
# Screenshot from 2020-05-22 23-28-32 Implemented my own image perturbation, best so far
# Screenshot from 2020-05-23 01-17-48 Same but half learning rate
# Screenshot from 2020-05-22 23-49-22 An additional 10 epochs after the above, overfitting
# Screenshot from 2020-05-23 00-32-06 batch of 32 and 1-9 pos-neg, terrible

# TODO
"""Progressively reduce the learning rate, with a scheduler or ReduceLROnPlateau
Verify what happens with proper distribution in validation set
Add required charts and stats
Find the right threashold as required
"""