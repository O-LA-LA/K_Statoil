import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

train = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/Train/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
print('train has been read')


# split icebergs from ships
# Index([995, 118, 135, 138]
icebergs = train[train.is_iceberg == 1].sample(n=4, random_state=38)

# Index([1276, 997, 181, 345]
ships = train[train.is_iceberg == 0].sample(n=4, random_state=38)

# creating a df for training the model
df = pd.concat([icebergs, ships])

# Defining anything under mean+std as noise and reduce it the minimum value
for idx in df.index:
    img = pd.Series(df.band_1[idx])
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)
    img[img < (img.mean()+img.std())] = 0
    df.band_1.ix[idx] = img.values.tolist()
    img = pd.Series(df.band_2[idx])
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)
    img[img < (img.mean()+img.std())] = 0
    df.band_2.ix[idx] = img.values.tolist()

# expanding the image from a list of pixels to individual columns
df[list(range(5625))] = pd.DataFrame(df.band_1.values.tolist(), index=df.index)
df[list(range(5625, (5625+5625), 1))] = pd.DataFrame(df.band_1.values.tolist(), index=df.index)

# cleaning the features to incluse only numerical data
X = df.drop(['band_1', 'band_2', 'is_iceberg', 'id'], axis = 1)

# isolating the target data
y = df['is_iceberg']

# split train and validate data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

############ from here on it's just copy paste from the mnist model. I dont realy know what is going on.

# change train to float32 for better performance
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Setup model parameters
batch_size = 2
num_classes = 1
epochs = 3

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])