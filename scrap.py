
from numpy import newaxis
import copy
from scipy import ndimage
import cv2 as cv
import numpy as np
import tensorflow as tf

# An image clearing dependencies
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)
from skimage.filters import gaussian
from skimage.color import rgb2gray

# Data reading and visualization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Training part
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator


def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])

def smooth(X, sigma):
    return np.asarray([gaussian(item, sigma=sigma) for item in X])

def grayscale(X):
    return np.asarray([rgb2gray(item) for item in X])

def create_dataset(frame, labeled, smooth_rgb=0.2, smooth_gray=0.5,
                   weight_rgb=0.05, weight_gray=0.05):
    band_1, band_2, images = frame['band_1'].values, frame['band_2'].values, color_composite(frame)
    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])
    band_1 = to_arr(band_1)
    band_2 = to_arr(band_2)
    band_3 = (band_1 + band_2) / 2
    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])
    # Make a picture format from flat vector
    band_1 = gray_reshape(band_1)
    band_2 = gray_reshape(band_2)
    band_3 = gray_reshape(band_3)
    print('Denoising and reshaping')
    if train_b and clean_b:
        # Smooth and denoise data
        band_1 = smooth(denoise(band_1, weight_gray, False), smooth_gray)
        print('Gray 1 done')
        band_2 = smooth(denoise(band_2, weight_gray, False), smooth_gray)
        print('Gray 2 done')
        band_3 = smooth(denoise(band_3, weight_gray, False), smooth_gray)
        print('Gray 3 done')
    if train_img and clean_img:
        images = smooth(denoise(images, weight_rgb, True), smooth_rgb)
    print('RGB done')
    tf_reshape = lambda x: np.asarray([item.reshape(75, 75, 1) for item in x])
    band_1 = tf_reshape(band_1)
    band_2 = tf_reshape(band_2)
    band_3 = tf_reshape(band_3)
    #images = tf_reshape(images)
    band = np.concatenate([band_1, band_2, band_3], axis=3)
    if labeled:
        y = np.array(frame["is_iceberg"])
    else:
        y = None
    return y, band, images

def get_model_notebook(lr, decay, channels, relu_type='relu'):
    # angle variable defines if we should use angle parameter or ignore it
    input_1 = Input(shape=(75, 75, channels))

    fcnn = Conv2D(32, kernel_size=(3, 3), activation=relu_type)(
        BatchNormalization()(input_1))
    fcnn = MaxPooling2D((3, 3))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = Conv2D(64, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = BatchNormalization()(fcnn)
    fcnn = Flatten()(fcnn)
    local_input = input_1
    partial_model = Model(input_1, fcnn)
    dense = Dropout(0.2)(fcnn)
    dense = Dense(256, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    # For some reason i've decided not to normalize angle data
    output = Dense(1, activation="sigmoid")(dense)
    model = Model(local_input, output)
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, partial_model

def combined_model(m_b, m_img, lr, decay):
    input_b = Input(shape=(75, 75, layers))
    input_img = Input(shape=(75, 75, layers))

    # I've never tested non-trainable source models tho
    #for layer in m_b.layers:
    #    layer.trainable = False
    #for layer in m_img.layers:
    #    layer.trainable = False

    m1 = m_b(input_b)
    m2 = m_img(input_img)

    # So, combine models and train perceptron based on that
    # The iteresting idea is to use XGB for this task, but i actually hate this method
    common = Concatenate()([m1, m2])
    common = BatchNormalization()(common)
    common = Dropout(0.3)(common)
    common = Dense(1024, activation='relu')(common)
    common = Dropout(0.3)(common)
    common = Dense(512, activation='relu')(common)
    common = Dropout(0.3)(common)
    output = Dense(1, activation="sigmoid")(common)
    model = Model([input_b, input_img], output)
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def gen_flow_multi_inputs(I1, I2, y, batch_size):
    gen1 = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             rotation_range=10)
    gen2 = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             rotation_range=10)
    genI1 = gen1.flow(I1, y, batch_size=batch_size, seed=57, shuffle=False)
    genI2 = gen2.flow(I1, I2, batch_size=batch_size, seed=57, shuffle=False)
    while True:
        I1i = genI1.next()
        I2i = genI2.next()
        #print I1i[0].shape
        np.testing.assert_array_equal(I2i[0], I1i[0])
        yield [I1i[0], I2i[1]], I1i[1]

def train_model(model, batch_size, epochs, checkpoint_name, X_train, y_train, val_data, verbose=2):
    callbacks = [ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.,
                                   height_shift_range=0.,
                                   channel_shift_range=0,
                                   zoom_range=0.2,
                                   rotation_range=10)
    x_test, y_test = val_data
    try:
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs,
                                    steps_per_epoch=len(X_train) / batch_size,
                                    validation_data=(x_test, y_test), verbose=1,
                                    callbacks=callbacks)
    except KeyboardInterrupt:
        if verbose > 0:
            print('Interrupted')
    if verbose > 0:
        print('Loading model')
    model.load_weights(filepath=checkpoint_name)
    return model

def gen_model_weights(lr, decay, channels, relu, batch_size, epochs, path_name, data, verbose=2):
    X_train, y_train, X_test, y_test, X_val, y_val = data
    model, partial_model = get_model_notebook(lr, decay, channels, relu)
    model = train_model(model, batch_size, epochs, path_name,
                           X_train, y_train, (X_test, y_test), verbose=verbose)

    if verbose > 0:
        loss_val, acc_val = model.evaluate(X_val, y_val,
                               verbose=0, batch_size=batch_size)

        loss_train, acc_train = model.evaluate(X_test, y_test,
                                       verbose=0, batch_size=batch_size)

        print('Val/Train Loss:', str(loss_val) + '/' + str(loss_train), \
            'Val/Train Acc:', str(acc_val) + '/' + str(acc_train))
    return model, partial_model

# Train all 3 models
def train_models(dataset, lr, batch_size, max_epoch, verbose=2, return_model=False):
    y_train, X_b, X_images = dataset
    y_train_full, y_val,\
    X_b_full, X_b_val,\
    X_images_full, X_images_val = train_test_split(y_train, X_b, X_images, random_state=687, train_size=0.9)

    y_train, y_test, \
    X_b_train, X_b_test, \
    X_images_train, X_images_test = train_test_split(y_train_full, X_b_full, X_images_full, random_state=576, train_size=0.85)

    if train_b:
        if verbose > 0:
            print('Training bandwidth network')
        data_b1 = (X_b_train, y_train, X_b_test, y_test, X_b_val, y_val)
        model_b, model_b_cut = gen_model_weights(lr, 1e-6, layers, 'relu', batch_size, max_epoch, 'model_b',
                                                 data=data_b1, verbose=verbose)

    if train_img:
        if verbose > 0:
            print('Training image network')
        data_images = (X_images_train, y_train, X_images_test, y_test, X_images_val, y_val)
        model_images, model_images_cut = gen_model_weights(lr, 1e-6, layers, 'relu', batch_size, max_epoch, 'model_img',
                                                       data_images, verbose=verbose)

    if train_total:
        common_model = combined_model(model_b_cut, model_images_cut, lr/2, 1e-7)
        common_x_train = [X_b_full, X_images_full]
        common_y_train = y_train_full
        common_x_val = [X_b_val, X_images_val]
        common_y_val = y_val
        if verbose > 0:
            print('Training common network')
        callbacks = [ModelCheckpoint('common', save_best_only=True, monitor='val_loss')]
        try:
            common_model.fit_generator(gen_flow_multi_inputs(X_b_full, X_images_full, y_train_full, batch_size),
                                         epochs=max_epoch,
                                  steps_per_epoch=len(X_b_full) / batch_size,
                                  validation_data=(common_x_val, common_y_val), verbose=1,
                                  callbacks=callbacks)
        except KeyboardInterrupt:
            pass
        common_model.load_weights(filepath='common')
        loss_val, acc_val = common_model.evaluate(common_x_val, common_y_val,
                                           verbose=0, batch_size=batch_size)
        loss_train, acc_train = common_model.evaluate(common_x_train, common_y_train,
                                                  verbose=0, batch_size=batch_size)
        if verbose > 0:
            print('Loss:', loss_val, 'Acc:', acc_val)
    if return_model:
        return common_model
    else:
        return (loss_train, acc_train), (loss_val, acc_val)


def my_tr_tst_split (fet1, fet2, fet3, fet4, tar, split=0.2, random_state=1):
    global bnd1_train, bnd1_val, bnd2_train, bnd2_val, fbnd1_train, fbnd1_val
    global fbnd2_train, fbnd2_val, tar_train, tar_val
    # confirm data length matches
    assert len(fet1) == len(tar) and len(fet2) == len(fet1) and len(fet2) == len(fet3)
    end = int(len(fet1)*split)

    # shuffel the arrays to evenly distribute the data
    fet1, fet2, fet3, fet4, tar = unison_shuffled_copies(fet1, fet2, fet3, fet4, tar, seed=random_state)

    bnd1_train = fet1[end:]
    bnd1_val = fet1[:end]

    bnd2_train = fet2[end:]
    bnd2_val = fet2[:end]

    fbnd1_train = fet3[end:]
    fbnd1_val = fet3[:end]

    fbnd2_train = fet4[end:]
    fbnd2_val = fet4[:end]

    tar_train = tar[end:]
    tar_val = tar[:end]

def add_axis (ff):
    ff = np.array(ff)
    ff = ff[:, :, newaxis]
    return ff

def rotate_img(img, rotate):
    ver_img = [img]
    rows, cols, layers = img.shape
    for angle in np.arange(0, 360, 360 / (rotate - 1)):
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        ver_img.extend([cv.warpAffine(img, M, (cols, rows))])
    return ver_img

def unison_shuffled_copies(a, b, c, d, e, seed):
    assert len(a) == len(b) and len(c) == len(b) and len(c) == len(d) and len(c) == len(e)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p], c.iloc[p], d.iloc[p], e[p]

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

def normalize_bd1 (img1):
    img1 = np.array(img1)
    # min of all band 1 excluding outlayers
    mn = -40
    # max of all band 1 excluding outlayers
    mx = 29
    img1 = (img1 - mn) / (mx - mn)
    return img1.reshape(75, 75)

def normalize_bd2 (img1):
    img1 = np.array(img1)
    # min of all band 2 excluding outlayers
    mn = -43
    # max of all band 2 excluding outlayers
    mx = 12
    img1 = (img1 - mn) / (mx - mn)
    return img1.reshape(75, 75)

def rotndup_img (img, num):
    ver_img = [img]
    for i in range(num-1):
        ver_img.extend([np.array(list(zip(*ver_img[i][::-1])))])
    return ver_img

def deep_clean (bd1):
    global off_std
    # simple noise cleaning, anything below mean plus 1 std will be considered noise
    bd1[bd1 < (bd1.mean() + bd1.std() * (off_std))] = 0

    # find length and width of image
    lng, wdth = list(bd1.shape)

    # defining an array to hold "remember" our object pattern
    part_of = np.zeros((lng, wdth))

    # we are searching waterfall style only upside down, and so we extract the peaks and work our way down
    # until we reach the previous pixel marked as 0. this will define the major structurs in the image.
    # All the rest is squashed.
    top99 = np.percentile(bd1, 99)
    row, col = np.where(bd1 > top99)

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

def prep_data(df, order):
    #avg = 0
    avg_inc = df.inc_angle.mean()
    max_inc = df.inc_angle.max()
    min_inc = df.inc_angle.min()
    inc_ang = np.zeros((int(75*order), int(75*order)))
    # x1 = preprocessed img with 2 layers of band_1
    x1 = []
    # x2 = preprocessed img with 2 layers of band_2
    x2 = []
    # x3 = feature data of band1: top 20 bars of hist, size of object, angle
    x3 = pd.DataFrame()
    # x4 = feature data of band2: top 20 bars of hist, size of object, angle
    x4 = pd.DataFrame()

    for i in df.index:
        print(i)
        # raw images
        if order != 1:
            rw1 = enlarge(normalize_bd1(df.at[i, 'band_1']), order=order)
            rw2 = enlarge(normalize_bd2(df.at[i, 'band_2']), order=order)
        else:
            rw1 = normalize_bd1(df.at[i, 'band_1'])
            rw2 = normalize_bd2(df.at[i, 'band_2'])

        # Cleaned layer band_1 HH (cHH)
        c_bnd1 = copy.deepcopy(rw1)
        c_bnd1 = deep_clean(c_bnd1)

        # Cleaned layer band_2 HV (cHV)
        c_bnd2 = copy.deepcopy(rw1)
        c_bnd2 = deep_clean(c_bnd2)

        # Incident angle as a normalized layer
        inc_ang[:] = ((df.at[i, 'inc_angle'] - avg_inc - min_inc) / (max_inc - min_inc))

        # 2 layers containing normalized raw and a clean layer
        x1.append(np.dstack((rw1, c_bnd1, inc_ang)))
        x2.append(np.dstack((rw2, c_bnd2, inc_ang)))

        x3.loc[i, 'inc_angle'] = df.loc[i, 'inc_angle']
        x3.loc[i, 'ob1_size'] = sum(sum(c_bnd1>0))
        x4.loc[i, 'inc_angle'] = df.loc[i, 'inc_angle']
        x4.loc[i, 'obj2_size'] = sum(sum(c_bnd2 > 0))

        for k in range(1, 51, 5):
            for bnd in ['band_1', 'band_2']:
                img = pd.DataFrame((df.at[i, bnd]), columns=['bck_s'])
                hist = np.histogram(img['bck_s'], bins=100)
                cl_nm1 = bnd + '_pxc_' + str(k)
                cl_nm2 = bnd + '_bck_' + str(k)
                if bnd == 'band_1':
                    x3.loc[i, cl_nm1] = sum(hist[0][k + 49: 5])
                    x3.loc[i, cl_nm2] = sum(hist[1][k + 50: 5])
                else:
                    x4.loc[i, cl_nm1] = sum(hist[0][k + 49: 5])
                    x4.loc[i, cl_nm2] = sum(hist[1][k + 50: 5])

    return np.array(x1), np.array(x2), x3, x4

def dup_data(fet1, fet2, fet3, fet4, tar, num = 4):

    # confirm data length matches
    assert len(fet1) == len(tar) and len(fet2) == len(fet3) and len(fet2) == len(fet4)
    x1 = []
    x2 = []
    x3 = pd.DataFrame(columns=fet3.columns)
    x4 = pd.DataFrame(columns=fet4.columns)
    y = []
    if num == 1:
        return fet1, fet2, fet3, fet4, tar
    else:
        for i in range(len(fet1)):
            print('duplicated image #', i)
            # rotate images
            x1.extend(list(rotate_img(fet1[i], rotate=num)))
            x2.extend(list(rotate_img(fet2[i], rotate=num)))
            y.extend([tar[i]] * num)
            for j in range(num):
                x3 = x3.append(fet3.iloc[i])
                x4 = x4.append(fet4.iloc[i])
        return np.array(x1), np.array(x2), x3, x4, np.array(y)

def import_training_data():
    global train
    # load the training data
    train = pd.read_json(path + 'Train/train.json')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

    # option 1 fill the images with nan for inc_angle with mean of angles
    train['inc_angle'] = train['inc_angle'].fillna(train['inc_angle'].mean())
    #test['inc_angle'] = test['inc_angle'].fillna(test['inc_angle'].mean())

    # option 2 drop images with nan
    #train = train.dropna(axis=0, how='any')

    print('train data has been read')

def data_wrangling(order, flip):
    global train, bnd1_tr, bnd2_tr, fbnd1_tr, fbnd2_tr, y_tr, bnd1_dup, bnd2_dup, fbnd1_dup, fbnd2_dup, y_dup

    bnd1_tr, bnd2_tr, fbnd1_tr, fbnd2_tr = prep_data(train, order=order)
    y_tr = np.array(train['is_iceberg'])

    # duplicating the data by rotating 90 degree 4 times:
    bnd1_dup, bnd2_dup, fbnd1_dup, fbnd2_dup, y_dup = dup_data(bnd1_tr, bnd2_tr, fbnd1_tr, fbnd2_tr, y_tr, num=flip)

def setup_input_shape(order, layer):
    global bnd1_train, bnd1_val, bnd2_train, bnd2_val, fbnd1_train, fbnd1_val, fbnd2_train, fbnd2_val, tar_train, tar_val, input_shape
    # input image dimensions
    img_rows, img_cols, layers = int(75 * order), int(75 * order), layer

    fbnd1_train = add_axis(fbnd1_train)
    fbnd1_val = add_axis(fbnd1_val)

    fbnd2_train = add_axis(fbnd2_train)
    fbnd2_val = add_axis(fbnd2_val)

    # configuring the input_shape and the x, y to the tensor flow structure
    if K.image_data_format() == 'channels_first':
        bnd1_train = bnd1_train.reshape(bnd1_train.shape[0], layers, img_rows, img_cols)
        bnd2_train = bnd2_train.reshape(bnd2_train.shape[0], layers, img_rows, img_cols)
        bnd1_val = bnd1_val.reshape(bnd1_val.shape[0], layers, img_rows, img_cols)
        bnd2_val = bnd2_val.reshape(bnd2_val.shape[0], layers, img_rows, img_cols)
        input_shape = (layers, img_rows, img_cols)
    else:
        bnd1_train = bnd1_train.reshape(bnd1_train.shape[0], img_rows, img_cols, layers)
        bnd2_train = bnd2_train.reshape(bnd2_train.shape[0], img_rows, img_cols, layers)
        bnd1_val = bnd1_val.reshape(bnd1_val.shape[0], img_rows, img_cols, layers)
        bnd2_val = bnd2_val.reshape(bnd2_val.shape[0], img_rows, img_cols, layers)
        input_shape = (img_rows, img_cols, layers)

def prep_submission():
    global X1_tst, X2_tst, X3_tst, X4_tst
    test = pd.read_json(path+'Test/test.json')
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

    # option2 drop images with nan
    test = test.dropna(axis=0, how='any')
    print('test has been read')

    X1_tst, X2_tst, X3_tst, X4_tst = prep_data(test, order)

    pred_test = common_model.predict([X1_tst, X2_tst])

    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

    scr = str(score[0]).replace(".", "")[0:5]

    submission.to_csv('submission_scr_{}.csv'.format(scr), index=False)


# Random initialization
np.random.seed(98643)
tf.set_random_seed(683)
# Uncomment this to hide TF warnings about allocation
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# Order to enlarge our images
order = 1
# number of layer per image
layers = 3
# number of duplicate images
duplicate = 2
# how far off the mean to clean when deep cleaning measured in STD
off_std = 1

batch_size = 50
epochs = 100

#path = '/data/'
path = '/Users/ohad/Google_Drive/DS_projects/K_Statoil/'

import_training_data()

data_wrangling(order, duplicate)

# split train and validate data
my_tr_tst_split(bnd1_dup, bnd2_dup, fbnd1_dup, fbnd2_dup, y_dup, split=0.2, random_state=30)

setup_input_shape(order, layers)

train_all = True

# These are train flags that required to train model more efficiently and
# select proper model parameters
train_b = True or train_all
train_img = True or train_all
train_total = True or train_all
predict_submission = True and train_all

clean_all = False
clean_b = False or clean_all
clean_img = False or clean_all

load_all = False
load_b = False or load_all
load_img = False or load_all


common_model = train_models((tar_train, bnd1_train, bnd2_train), 7e-04, 32, 50, 1, return_model=True)



score = common_model.evaluate([bnd1_val, bnd2_val], tar_val, verbose=1)
print('Test val loss:', score[0])
print('Test val accuracy:', score[1])

prep_submission()

if predict_submission:
    print('Reading test dataset')
    test = pd.read_json("../test/test.json")
    X_fin_b, X_fin_img, X3_tst, X4_tst = prep_data(test, 1)
    print('Predicting')
    prediction = common_model.predict([X_fin_b, X_fin_img], verbose=1, batch_size=32)
    print('Submitting')
    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})

    submission.to_csv("./submission_validation.csv", index=False)
    print('Done')


subm = pd.read_csv('./submission_scr_02028.csv', index_col=0)
subval = pd.read_csv('./submission_validation.csv', index_col=0)

subm['is_ice1'] = subval['is_iceberg']

sum(subm.is_iceberg == subval.is_iceberg) == len(subm)