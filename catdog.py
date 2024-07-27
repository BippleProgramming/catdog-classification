import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# gpus = tf.config.experimental.list_physical_devices('CPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# accessing folder named data and getting all those file extensions
data_dir = 'data' 
image_exts = ['jpeg','jpg', 'bmp', 'png']

# goes through happy then sad
for image_class in os.listdir(data_dir): 
    # goes through each image in folder
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path) # loads image and sees if it can be read
            tip = imghdr.what(image_path) # returns image type (jpg)
            if tip not in image_exts: # if tip not in image exts then it removes it
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: # scans for corrupted files, removes it and continues with code
            print('Issue with image {}'.format(image_path))
            os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory('data') # gets all data
data_iterator = data.as_numpy_iterator() # allows us to use numpy functions

batch = data_iterator.next() # getting batch size 32 (default)

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])

data = data.map(lambda x,y: (x/255, y)) # goes from 0-1 to 0-255 for format purposes
# data.as_numpy_iterator().next() 

# splitting batch into chunks
train_size = int(len(data)*.7) # 70%
val_size = int(len(data)*.2) # 20%
test_size = int(len(data)*.1) # 10%

# takes those chunks
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary() #summary
