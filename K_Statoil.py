import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def clean_img (img):
    img = np.array(img)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)
    img[img < (img.mean()+img.std()*1)] = 0
    return img.reshape((75, 75))

def normalize (img1):
    img1 = np.array(img1)
    mn, mx = img1.min(), img1.max()
    img1 = (img1 - mn) / (mx - mn)
    return img1.reshape(75, 75)

def rotndup_img (img):
    img90 = np.array(list(zip(*img[::-1])))
    img180 = np.array(list(zip(*img90[::-1])))
    img270 = np.array(list(zip(*img180[::-1])))
    return [img, img90, img180, img270]

def deep_clean (bd1):
    # begins as a regular cleaning of noise anything below mean + 1 std will zero
    bd1 = np.array(bd1)
    mn, mx = bd1.min(), bd1.max()
    bd1 = (bd1 - mn) / (mx - mn)
    bd1[bd1 < (bd1.mean() + bd1.std() * 1)] = 0
    bd1 = bd1.reshape(75, 75)

    # defining an array to hold "remember" our object pattern
    part_of = np.zeros((75, 75))

    # we are searching waterfall style only upside down, and so we extract the peaks and work our way down
    # until we reach the previous pixel marked as 0. this will define the major structurs in the image.
    # All the rest is squashed.
    top95 = np.percentile(bd1, 99.5)
    row, col = np.where(bd1 > top95)

    lst = [[-1, 1], [0, 1], [1, 1], [-1, 0], [1, 0], [-1, -1], [0, -1], [-1, -1]]

    for i in range(len(row)):
        x = row[i]
        y = col[i]
        cell2chk = [[x, y]]
        j = 0
        macktub = True
        while macktub:
            x, y = cell2chk[j]
            part_of[x, y] = 1
            for sx, sy in lst:
                if (x+sx) < 75 and (y+sy) < 75 and (x+sx) >= 0 and (y+sy) >= 0:
                    if bd1[x + sx, y + sy] != 0 and part_of[x + sx, y + sy] == 0:
                        part_of[x + sx, y + sy] = 1
                        cell2chk.append([x + sx, y + sy])
                    elif bd1[x + sx, y + sy] == 0:
                        part_of[x + sx, y + sy] = -1
            j += 1
            if j + 1 >= len(cell2chk):
                macktub = False

    bd1[part_of == 0] = 0
    return bd1


def getModel():
    # Build keras model

    model = Sequential()

    # CNN 1

    model.add(Conv2D(256, kernel_size=(3, 3), input_shape=input_shape))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
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

    optimizer = Adam(lr= 0.0011, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #optimizer = Adam(lr=0.001, decay=0.0)
    #optimizer = keras.optimizers.SGD(lr=0.03, momentum=0.03, decay=0.01, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# load the training data
train = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/Train/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

# option1 fill the images with nan for inc_angle with mean of angles
#train['inc_angle'] = train['inc_angle'].fillna(train['inc_angle'].mean())
#test['inc_angle'] = test['inc_angle'].fillna(test['inc_angle'].mean())

# option2 drop images with nan
train = train.dropna(axis=0, how='any')


print('train has been read')

# split icebergs from ships
# Index([995, 118, 135, 138]
"""icebergs = train[train.is_iceberg == 1].sample(n=4, random_state=38)

# Index([1276, 997, 181, 345]
ships = train[train.is_iceberg == 0].sample(n=4, random_state=38)

# creating a df for training the model
df = pd.concat([icebergs, ships])"""

# input image dimensions
img_rows, img_cols, layers = 75, 75, 4


# train var setup
avg = 0
avg_inc = train.inc_angle.mean()
max_inc = train.inc_angle.max()
min_inc = train.inc_angle.min()
rw1_ly = []
rw2_ly = []
rrw_ly = []
enh_ly = []
bnd1_ly = []
bnd2_ly = []
inc_ang_ly = []
inc_ang = np.zeros((75, 75))
X = []
y = []

for i in train.index:
    print(i)

    """"# cleaned layer band_1 HH (cHH)
    bnd1 = clean_img(train.at[i, 'band_1'])
    bnd1_ly.extend(rotndup_img(bnd1))

    # cleaned layer band_2 HV (in the future will cleaned based on the zero of HH) - cHV
    bnd2 = clean_img(train.at[i, 'band_2'])
    bnd2_ly.extend((rotndup_img(bnd2)))"""

    # train raw images
    rw1 = normalize(train.at[i, 'band_1'])
    rw2 = normalize(train.at[i, 'band_2'])

    # train cleaned layer band_1 HH (cHH)
    bnd1 = deep_clean(train.at[i, 'band_1'])

    # train cleaned layer band_2 HV (in the future will cleaned based on the zero of HH) - cHV
    bnd2 = np.array(train.at[i, 'band_2']).reshape(75, 75)
    bnd2[bnd1 == 0] = 0

    # train Enhanced layer of cHH + cHV
    enh_img = bnd1 + bnd2
    avg = avg + np.mean(enh_img, axis = (0, 1))


    # increasing sample data by rotating images
    #rw1_ly.extend(rotndup_img(rw1))
    #rw2_ly.extend(rotndup_img(rw2))
    #rrw_ly.extend(rotndup_img(rw1+rw2))
    #bnd1_ly.extend(rotndup_img(bnd1))
    #bnd2_ly.extend((rotndup_img(bnd2)))
    #enh_ly.extend(rotndup_img(enh_img))


    # layer containing the angle
    #inc_ang[bnd1 != 0] = train.at[i, 'inc_angle']
    #inc_ang[bnd1 == 0] = 0
    inc_ang[:] = ((train.at[i, 'inc_angle'] - avg_inc - min_inc)/ (max_inc - min_inc))
    #inc_ang_ly.extend([inc_ang, inc_ang, inc_ang, inc_ang])

    # matching the target data
    ice = train.loc[i, 'is_iceberg']

    """
    # All 4 layers bnd1, bnd2, enh and angle
    X.append(np.dstack((bnd1_ly[0], bnd2_ly[0], enh_ly[0], inc_ang_ly[0])))
    X.append(np.dstack((bnd1_ly[1], bnd2_ly[1], enh_ly[1], inc_ang_ly[1])))
    X.append(np.dstack((bnd1_ly[2], bnd2_ly[2], enh_ly[2], inc_ang_ly[2])))
    X.append(np.dstack((bnd1_ly[3], bnd2_ly[3], enh_ly[3], inc_ang_ly[3])))

    y.extend([ice, ice, ice, ice])
    """

    """
    # layer NOT containing the angle and the enh_ly
    X.append(np.dstack((bnd1_ly[0], bnd2_ly[0])))
    X.append(np.dstack((bnd1_ly[1], bnd2_ly[1])))
    X.append(np.dstack((bnd1_ly[2], bnd2_ly[2])))
    X.append(np.dstack((bnd1_ly[3], bnd2_ly[3])))
    
    y.extend([ice, ice, ice, ice])
    """

    """
    # layer bnd1, bnd2, angle
    X.append(np.dstack((bnd1_ly[0], bnd2_ly[0], inc_ang_ly[0])))
    X.append(np.dstack((bnd1_ly[1], bnd2_ly[1], inc_ang_ly[1])))
    X.append(np.dstack((bnd1_ly[2], bnd2_ly[2], inc_ang_ly[2])))
    X.append(np.dstack((bnd1_ly[3], bnd2_ly[3], inc_ang_ly[3])))

    y.extend([ice, ice, ice, ice])
    """
    """
    # layer containing ONLY enh + angle with no duplication of the data
    X.append(np.dstack((enh_img, inc_ang)))

    y.extend([ice])
    """

    """
    # layer containing ONLY enh with NO duplication of the data
    X.append(np.dstack(enh_img))
    
    y.extend([ice])
    """

    # layer containing enh and normalized raw with NO duplication of the data
    X.append(np.dstack((rw1 + rw2, enh_img, rw1, inc_ang)))

    y.extend([ice])

    """
    # layer containing enh and normalized raw with duplication of the data
    X.append(np.dstack((rrw_ly[0], rw1_ly[0], enh_ly[0], inc_ang)))
    X.append(np.dstack((rrw_ly[1], rw1_ly[1], enh_ly[1], inc_ang)))
    X.append(np.dstack((rrw_ly[2], rw1_ly[2], enh_ly[2], inc_ang)))
    X.append(np.dstack((rrw_ly[3], rw1_ly[3], enh_ly[3], inc_ang)))

    y.extend([ice, ice, ice, ice])
    """

    """
    # layer containing ONLY enh with duplication of the data
    X.append(np.dstack(enh_ly[0]))
    X.append(np.dstack(enh_ly[1]))
    X.append(np.dstack(enh_ly[2]))
    X.append(np.dstack(enh_ly[3]))

    y.extend([ice, ice, ice, ice])
    """
"""
what I want to do here is create 5 layers:
    1. Cleaned layer band_1
    2. Cleaned layer band_2
    3. Enhanced using band_1 + band_2
    4. layer using the angle
    5. to be tested - img from 4 zoomed, sharpened 
"""

"""
import scipy
from scipy import ndimage

o_bnd1 = train['band_1'].apply(np.array, axis=1).apply(np.reshape(75, 75))
n_bnd1 = np.zeros(shape=(150, 150), dtype=np.float32)
o_bnd2 = np.array(train['band_2'][0]).reshape(75, 75)
n_bnd2 = np.zeros(shape=(150, 150), dtype=np.float32)


for col in range(150):
    for row in range(150):
        n_bnd1[col, row] = o_bnd1[int(col/2), int(row/2)]
        n_bnd2[col, row] = o_bnd2[int(col/2), int(row/2)]

pwr = 2
pwr2 = 2

bn1_f = ndimage.gaussian_filter(n_bnd1, pwr)
f_bn1_f = ndimage.gaussian_filter(bn1_f, pwr2)

alpha = 30
shrp_bn1 = bn1_f + alpha * (bn1_f - f_bn1_f)

plt.subplot(231)
plt.imshow(n_bnd1[50:100, 55:80])
plt.title('org')
plt.subplot(232)
plt.imshow(bn1_f[50:100, 55:80])
plt.title('blur')
plt.subplot(233)
plt.imshow(shrp_bn1[50:100, 55:80])
plt.title('sharp')

bn2_f = ndimage.gaussian_filter(n_bnd2, pwr)
f_bn2_f = ndimage.gaussian_filter(bn2_f, pwr2)

alpha = 30
shrp_bn2 = bn2_f + alpha * (bn2_f - f_bn2_f)

plt.subplot(234)
plt.imshow(n_bnd2[50:100, 55:80])
plt.title('org')
plt.subplot(235)
plt.imshow(bn2_f[50:100, 55:80])
plt.title('blur')
plt.subplot(236)
plt.imshow(shrp_bn2[50:100, 55:80])
plt.title('sharp')

plt.tight_layout()
plt.show()
"""

X = np.array(X)
avg = avg/len(X)
#X = X - avg
y = np.array(y)

X, y = unison_shuffled_copies(X, y)

# split train and validate data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# configuring the input_shape and the x, y to the tensor flow structure
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], layers, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], layers, img_rows, img_cols)
    input_shape = (layers, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, layers)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, layers)
    input_shape = (img_rows, img_cols, layers)


# change train to float32 for better performance
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Setup model parameters
batch_size = 100
epochs = 50
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
print('Test val loss:', score[0])
print('Test val accuracy:', score[1])

"""
####  getting the data ready for submission  #####
test = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/test/test.json')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

# option2 drop images with nan
test = test.dropna(axis=0, how='any')
print('test has been read')


# test var setup

tst_avg_inc = test.inc_angle.mean()
tst_max_inc = test.inc_angle.max()
tst_min_inc = test.inc_angle.min()
tst_rw1_ly = []
tst_rw2_ly = []
tst_rrw_ly = []
tst_enh_ly = []
# tst_bnd1_ly = []
# tst_bnd2_ly = []
# tst_inc_ang_ly = []
tst_inc_ang = np.zeros((75, 75))
tst_X = []
tst_y = []


for i in test.index:
    print(i)

    # test raw images
    tst_rw1 = normalize(test.at[i, 'band_1'])
    tst_rw2 = normalize(test.at[i, 'band_2'])

    # test cleaned layer band_1 HH (cHH)
    tst_bnd1 = deep_clean(test.at[i, 'band_1'])

    # test cleaned layer band_2 HV (in the future will cleaned based on the zero of HH) - cHV
    tst_bnd2 = np.array(test.at[i, 'band_2']).reshape(75, 75)
    tst_bnd2[tst_bnd1 == 0] = 0

    # train Enhanced layer of cHH + cHV
    tst_enh_img = tst_bnd1 + tst_bnd2

    # test creating a layer from the angle
    tst_inc_ang[:] = ((test.at[i, 'inc_angle'] - avg_inc - min_inc) / (max_inc - min_inc))

    # layer containing enh and normalized raw with NO duplication of the data
    tst_X.append(np.dstack((tst_rw1 + tst_rw2, tst_enh_img, tst_rw1, tst_inc_ang)))

    ice = test.loc[i, 'is_iceberg']

    tst_y.extend([ice])

score = model.evaluate(tst_X, tst_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


"""