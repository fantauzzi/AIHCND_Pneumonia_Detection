#!/usr/bin/env python
# coding: utf-8

# ## Skeleton Code
# 
# The code below provides a skeleton for the model building & training component of your project. You can add/remove/build on code however you see fit, this is meant as a starting point.

# In[17]:


import numpy as np
import pandas as pd
import os
from glob import glob
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import preprocess_input
from itertools import chain
from math import ceil

batch_size = 16
validation_batch_size = 32
epochs = 5
augmentation_rate = 2

##Import any other stats/DL/ML packages you may need here. E.g. Keras, scikit-learn, etc.


# ## Do some early processing of your metadata for easier model training:

# In[18]:

# Hack to prevent TF from allocating the whole GPU memory at start-up
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

# Check there is a GPU available
n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
assert n_gpus >= 1

dataset_root = '/media/fanta/52A80B61A80B42C9/Users/fanta/datasets'
pd.set_option('display.max_rows', 100)

# In[19]:


all_xray_df = pd.read_csv(dataset_root + '/data/Data_Entry_2017.csv')
all_xray_df = all_xray_df.drop(columns=all_xray_df.columns[-1])
all_xray_df.sample(10)

# In[20]:


## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation

# all_xray_df = pd.read_csv(dataset_root+'/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join(dataset_root + '/data', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# In[21]:


## Here you may want to create some extra columns in your table with binary indicators of certain diseases 
## rather than working directly with the 'Finding Labels' column

# Todo
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for c_label in all_labels:
    all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)
all_xray_df.sample(10)


# In[ ]:


## Here we can create a new column called 'pneumonia_class' that will allow us to look at 
## images with or without pneumonia for binary classification

# Todo


# ## Create your training and testing data:

# In[ ]:


def create_splits(data):
    ## Either build your own or use a built-in library to split your original dataframe into two sets
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia

    # Todo

    train_data, val_data = train_test_split(data,
                                            test_size=.2,
                                            stratify=data['Pneumonia'],
                                            random_state=42,
                                            shuffle=True)

    return train_data, val_data


# # Now we can begin our model-building & training

# #### First suggestion: perform some image augmentation on your data

# In[ ]:


def my_image_augmentation(vargs):
    ## recommendation here to implement a package like Keras' ImageDataGenerator
    ## with some of the built-in augmentations 

    ## keep an eye out for types of augmentation that are or are not appropriate for medical imaging data
    ## Also keep in mind what sort of augmentation is or is not appropriate for testing vs validation data

    ## STAND-OUT SUGGESTION: implement some of your own custom augmentation that's *not*
    ## built into something like a Keras package

    # Todo

    return my_idg


def make_train_gen(train_df, dataset_root, batch_size):
    ## Create the actual generators using the output of my_image_augmentation for your training data
    ## Suggestion here to use the flow_from_dataframe library, e.g.:

    # TODO should I zero center the images? See https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input

    idg = ImageDataGenerator(horizontal_flip=True,  # rescale=1./255
                             vertical_flip=False,
                             height_shift_range=.0,
                             width_shift_range=.0,
                             rotation_range=10,
                             shear_range=0.1,
                             zoom_range=0.1,
                             preprocessing_function=preprocess_input)

    train_gen = idg.flow_from_dataframe(dataframe=train_df,
                                        directory=dataset_root,
                                        x_col='path',
                                        y_col='Pneumonia',
                                        class_mode='raw',  # TODO should I use binarty instead?
                                        target_size=(224, 224),  # Input size for VGG16
                                        batch_size=batch_size
                                        )

    return train_gen


def make_val_gen(val_df, dataset_root, batch_size):
    #     val_gen = my_val_idg.flow_from_dataframe(dataframe = val_data,
    #                                              directory=None,
    #                                              x_col = ,
    #                                              y_col = ',
    #                                              class_mode = 'binary',
    #                                              target_size = ,
    #                                              batch_size = )

    # Todo
    # idg = ImageDataGenerator(rescale=1. / 255.0)
    idg = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_gen = idg.flow_from_dataframe(dataframe=val_df,
                                      directory=dataset_root,
                                      x_col='path',
                                      y_col='Pneumonia',
                                      class_mode='raw',  # TODO should I use binarty instead?
                                      target_size=(224, 224),  # Input size for VGG16
                                      batch_size=batch_size
                                      )

    return val_gen


# In[ ]:

def enforce_classes_ratio(dataset_df, ratio):
    # Reduce the training set removing enough negative cases to remain with a training set with 50% positive and 50% negative
    count_train_pos = sum(dataset_df.Pneumonia)

    res_df = dataset_df[dataset_df.Pneumonia == 0][:int(count_train_pos * ratio)].append(
        dataset_df[dataset_df.Pneumonia == 1])
    res_df = shuffle(res_df)
    res_df.reset_index(inplace=True, drop=True)
    return res_df


train_df, test_df = create_splits(all_xray_df)
# Reduce the training set removing enough negative cases to remain with a training set with 50% positive and 50% negative

train_df = enforce_classes_ratio(train_df, 1)
test_df = enforce_classes_ratio(test_df, 3)

## May want to pull a single large batch of random validation data for testing after each epoch:
val_gen = make_val_gen(test_df, dataset_root, validation_batch_size)

# In[ ]:


## May want to look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, 
## and can be compared with how the raw data look prior to augmentation

train_gen = make_train_gen(train_df, dataset_root, batch_size)

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

# ## Build your model: 
# 
# Recommendation here to use a pre-trained network downloaded from Keras for fine-tuning

# In[ ]:

model = VGG16(include_top=True, weights='imagenet')
model.summary()

transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input,
                  outputs=transfer_layer.output)

for layer in vgg_model.layers[0:17]:
    layer.trainable = False

for layer in vgg_model.layers:
    print(layer.name, layer.trainable)

new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(vgg_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1, activation='sigmoid'))

## Set our optimizer, loss function, and learning rate
optimizer = Adam(lr=1e-4)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

new_model.summary()

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

steps_per_epoch = ceil(len(train_df) / batch_size) * augmentation_rate
# steps_per_epoch = 10
validation_steps = ceil(len(test_df) / validation_batch_size)
# validation_steps = 10
history = new_model.fit(x=train_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        epochs=epochs)


# Define a function here that will plot loss, val_loss, binary_accuracy, and val_binary_accuracy over all of
# your epochs:
def plot_history(history):
    n_epochs = len(history.history["loss"])
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
    ax1.set_xticks(range(n_epochs))
    ax1.plot(np.arange(0, n_epochs), history.history["loss"], label="Train. loss", color='greenyellow')
    ax1.plot(np.arange(0, n_epochs), history.history["val_loss"], label="Val. loss", color='darkolivegreen')

    ax2 = ax1.twinx()
    ax2.set_ylabel('F1')
    ax2.set_xticks(range(n_epochs))
    ax2.plot(np.arange(0, n_epochs), train_F1, label="Train. F1", color='magenta')
    ax2.plot(np.arange(0, n_epochs), val_F1, label="Val. F1", color='darkmagenta')

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

    return my_model


## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
## to output class activation maps to aid in clinical interpretation of your model's results


# In[ ]:


## Below is some helper code that will allow you to add checkpoints to your model,
## This will save the 'best' version of your model by comparing it to previous epochs of training

## Note that you need to choose which metric to monitor for your model's 'best' performance if using this code. 
## The 'patience' parameter is set to 10, meaning that your model will train for ten epochs without seeing
## improvement before quitting

# Todo

# weight_path="{}_my_model.best.hdf5".format('xray_class')

# checkpoint = ModelCheckpoint(weight_path, 
#                              monitor= CHOOSE_METRIC_TO_MONITOR_FOR_PERFORMANCE, 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC, 
#                              save_weights_only = True)

# early = EarlyStopping(monitor= SAME_AS_METRIC_CHOSEN_ABOVE, 
#                       mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC, 
#                       patience=10)

# callbacks_list = [checkpoint, early]


# ### Start training! 

# In[ ]:


## train your model

# Todo

# history = my_model.fit_generator(train_gen, 
#                           validation_data = (valX, valY), 
#                           epochs = , 
#                           callbacks = callbacks_list)


# ##### After training for some time, look at the performance of your model by plotting some performance statistics:
# 
# Note, these figures will come in handy for your FDA documentation later in the project

# In[ ]:


## After training, make some predictions to assess your model's overall performance
## Note that detecting pneumonia is hard even for trained expert radiologists, 
## so there is no need to make the model perfect.
my_model.load_weights(weight_path)
pred_Y = new_model.predict(valX, batch_size=32, verbose=True)


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

model_json = my_model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)
