import numpy as np
import pandas as pd
import random
from scipy import ndimage
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
from vgg16 import VGG16
from imagenet_utils import preprocess_input, decode_predictions

############################################################
############### All Functions ##############################
############################################################

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def enlarge (img, order=2):
    lng, wdth = list(img.shape)
    lng = int(lng*order)
    wdth = int(wdth*order)
    n_img = np.zeros(shape=(lng, wdth), dtype=np.float32)

    for col in range(lng):
        for row in range(wdth):
            n_img[col, row] = img[int(col/order), int(row/order)]
    return ndimage.gaussian_filter(n_img, 2)

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

def rotndup_img (img, num):
    ver_img = [img]
    for i in range(num-1):
        ver_img.extend([np.array(list(zip(*ver_img[i][::-1])))])
    return ver_img

def deep_clean (bd1):
    # simple noise cleaning, anything below mean plus 1 std will be considered noise
    bd1[bd1 < (bd1.mean() + bd1.std() * 1)] = 0

    # find length and width of image
    lng, wdth = list(bd1.shape)

    # defining an array to hold "remember" our object pattern
    part_of = np.zeros((lng, wdth))

    # we are searching waterfall style only upside down, and so we extract the peaks and work our way down
    # until we reach the previous pixel marked as 0. this will define the major structurs in the image.
    # All the rest is squashed.
    top95 = np.percentile(bd1, 99)
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
                if (x+sx) < wdth and (y+sy) < lng and (x+sx + y+sy) >= 0:
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

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

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

    optimizer = Adam(lr= 0.002, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.01)
    #optimizer = Adam(lr=0.001, decay=0.0)
    #optimizer = keras.optimizers.SGD(lr=0.03, momentum=0.03, decay=0.01, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def prep_data(df, order):
    #avg = 0
    avg_inc = df.inc_angle.mean()
    max_inc = df.inc_angle.max()
    min_inc = df.inc_angle.min()
    inc_ang = np.zeros((int(75*order), int(75*order)))
    X = []
    for i in df.index:
        print(i)
        # raw images
        rw1 = enlarge(normalize(df.at[i, 'band_1']), order=order)
        rw2 = enlarge(normalize(df.at[i, 'band_2']), order=order)

        # Cleaned layer band_1 HH (cHH)
        c_bnd1 = deep_clean(rw1)

        # Clean band_2 HV using cleaned band_1 - cHV
        c_bnd2 = rw2
        c_bnd2[c_bnd1 == 0] = 0

        # Enhanced layer of cHH + cHV
        enh_img = c_bnd1 + c_bnd2
        #avg = avg + np.mean(enh_img, axis=(0, 1))

        # Incident angle as a normalized layer
        inc_ang[:] = ((df.at[i, 'inc_angle'] - avg_inc - min_inc) / (max_inc - min_inc))

        # 5 layers containing enh and normalized raw with NO duplication of the data
        X.append(np.dstack((rw1, c_bnd1, rw2, c_bnd2, inc_ang)))

    return np.array(X)

def dup_data(fet, tar, num = 4):

    # confirm data length matches
    assert len(fet) == len(tar)
    X = []
    y = []
    for i in range(len(fet)):
        print('duplicated image #', i)
        # rotate images
        X.extend(list(rotndup_img(fet[i], num=num)))
        y.extend([tar[i]] * num)
    return np.array(X), np.array(y)

############################################################
########### Import Training data ###########################
############################################################

# load the training data
train = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/Train/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

# option1 fill the images with nan for inc_angle with mean of angles
#train['inc_angle'] = train['inc_angle'].fillna(train['inc_angle'].mean())
#test['inc_angle'] = test['inc_angle'].fillna(test['inc_angle'].mean())

# option2 drop images with nan
train = train.dropna(axis=0, how='any')

print('train has been read')

############################################################
########### Preparing data for model #######################
############################################################

# Order to enlarge our images
order = 1.2

# input image dimensions
img_rows, img_cols, layers = int(75 * order), int(75 * order), 5

X_tr = prep_data(train, order=order)
y_tr = np.array(train['is_iceberg'])

# duplicating the data by rotating 90 degree 4 times:
X_tr, y_tr = dup_data(X_tr, y_tr, num=2)

# shuffel the arrays to evenly distribute the data
X, y = unison_shuffled_copies(X_tr, y_tr)

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

############################################################
###########  model setup  ##################################
############################################################

# Setup model parameters
batch_size = 20
epochs = 50
earlyStopping = EarlyStopping(monitor='val_loss', patience=13, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model = getModel()
model.summary()

############################################################
###########  run model   ###################################
############################################################


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


############################################################
############  getting data ready for submission  ###########
############################################################


test = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/test/test.json')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

# option2 drop images with nan
test = test.dropna(axis=0, how='any')
print('test has been read')

X_tst = prep_data_nodup(test)

pred_test = model.predict(X_tst)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

scr =str(score[0]).replace(".", "")[0:5]

submission.to_csv('submission_scr_{}.csv'.format(scr), index=False)


"""
########################################################
###### Draft # Draft # Draft # Draft# Draft# Draft #####
########################################################

import scipy
from scipy import ndimage

o_bnd1 = normalize(np.array(train['band_1'][0]).reshape(75, 75))
n_bnd1 = np.zeros(shape=(150, 150), dtype=np.float32)
o_bnd2 = normalize(np.array(train['band_2'][0]).reshape(75, 75))
n_bnd2 = np.zeros(shape=(150, 150), dtype=np.float32)


for col in range(150):
    for row in range(150):
        n_bnd1[col, row] = o_bnd1[int(col/2), int(row/2)]
        n_bnd2[col, row] = o_bnd2[int(col/2), int(row/2)]

pwr = 1
pwr2 = 3

bn1_f = ndimage.gaussian_filter(n_bnd1, pwr)
f_bn1_f = ndimage.gaussian_filter(bn1_f, pwr2)

plt.subplot(121)
plt.imshow(bn1_f)
plt.title('bn1_f')
plt.subplot(122)
plt.imshow(f_bn1_f)



alpha = 0
shrp_bn1 = bn1_f + alpha * (bn1_f - f_bn1_f)

plt.subplot(221)
plt.imshow(n_bnd1)
plt.title('org')
plt.subplot(222)
plt.imshow(bn1_f)
plt.title('blur')
plt.subplot(223)
plt.imshow(shrp_bn1)
plt.title('sharp')
plt.subplot(224)
plt.imshow(o_bnd1)
plt.title('old')


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


"""