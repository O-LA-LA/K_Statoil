import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam


def getModel():
    # Build keras model

    model = Sequential()

    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr= 0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


train = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/Train/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
print('train has been read')


# split icebergs from ships
# Index([995, 118, 135, 138]
"""icebergs = train[train.is_iceberg == 1].sample(n=4, random_state=38)

# Index([1276, 997, 181, 345]
ships = train[train.is_iceberg == 0].sample(n=4, random_state=38)

# creating a df for training the model
df = pd.concat([icebergs, ships])"""

# Defining anything under mean+std as noise and reduce it the minimum value

def clean_img (img):
    img = np.array(img)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)
    img[img < (img.mean()+img.std())] = 0
    return img.tolist()

train['band_1'] = train['band_1'].apply(clean_img)
train['band_2'] = train['band_2'].apply(clean_img)


# adding the cleaned images to increase gradient of shapes and to reduce the amount of data to process and reshape to 75, 75
"""for idx in df.index:
    df.at[idx, 'enh_img'] = (np.array(df.loc[idx, 'band_1']) + np.array(df.loc[idx,'band_2'])).reshape(75,75)

df1 = pd.DataFrame([(np.array(df.at[idx, 'band_1']) + np.array(df.at[idx,'band_2'])).reshape(75,75) for idx in df.index], index=df.index)
for idx in df.index:
    df1[idx] = (np.array(df.at[idx, 'band_1']) + np.array(df.at[idx,'band_2'])).reshape(75,75)"""

#arr = np.array([np.array(df.at[idx,'band_2']).reshape(75, 75).tolist() + np.array(df.at[idx,'band_1']).reshape(75, 75).tolist() for idx in df.index])
#arr = np.array([np.array((df.at[idx,'band_2']) + np.array(df.at[idx,'band_1'])).reshape(75, 75) for idx in df.index])

# input image dimensions
img_rows, img_cols = 75, 75


# Creating more test cases
print('creating more test cases')

X = []
y = []
for i in train.index:
    print(i)
    img = np.array(np.array(train.at[i, 'band_2']) + np.array(train.at[i, 'band_1'])).reshape(75, 75)
    img90 = np.array(list(zip(*img[::-1])))
    img180 = np.array(list(zip(*img90[::-1])))
    img270 = np.array(list(zip(*img180[::-1])))
    X.extend([img, img90, img180, img270])
    ice = train.loc[i, 'is_iceberg']
    y.extend([ice, ice, ice, ice])

X = np.array(X)
y = np.array(y)

# split train and validate data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# configuring the input_shape and the x, y to the tensor flow structure
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# change train to float32 for better performance
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Setup model parameters
batch_size = 50
epochs = 3
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model = getModel()
model.summary()


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=(x_test, y_test))

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

