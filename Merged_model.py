
from keras.layers import Merge, Input, Concatenate
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
import sys
import time
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import Adam, RMSprop
import cv2 as cv

############################################################
############### All Functions ##############################
############################################################

def replace_cmd_line(output):
    """Replace the last command line output with the given output."""
    #sys.stdout.write(output + '\r')
    #sys.stdout.flush()
    #sys.stdout.write('\r')
    #sys.stdout.flush()
    print(output, end='')
    print('\r', end='')

def my_tr_tst_split (fet1, fet2, tar, split=0.2, random_state=1):
    # confirm data length matches
    assert len(fet1) == len(tar) and len(fet2) == len(fet1)
    end = int(len(fet1)*split)

    # shuffel the arrays to evenly distribute the data
    fet1, fet2, tar = unison_shuffled_copies(fet1, fet2, tar, seed=random_state)

    fet1_train = fet1[end:]
    fet1_val = fet1[:end]

    fet2_train = fet2[end:]
    fet2_val = fet2[:end]

    tar_train = tar[end:]
    tar_val = tar[:end]

    return fet1_train, fet1_val, fet2_train, fet2_val, tar_train, tar_val

def rotate_img(img, rotate):
    ver_img = [img]
    rows, cols, layers = img.shape
    for angle in np.arange(0, 360, 360 / (rotate - 1)):
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        ver_img.extend([cv.warpAffine(img, M, (cols, rows))])
    return ver_img

def unison_shuffled_copies(a, b, c, seed):
    assert len(a) == len(b) and len(c) == len(b)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

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

def prep_data(df, order):
    #avg = 0
    avg_inc = df.inc_angle.mean()
    max_inc = df.inc_angle.max()
    min_inc = df.inc_angle.min()
    inc_ang = np.zeros((int(75*order), int(75*order)))
    X1 = []
    X2 = []
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
        c_bnd1 = deep_clean(rw1)

        # Clean band_2 HV using cleaned band_1 - cHV
        #c_bnd2 = rw2
        #c_bnd2[c_bnd1 == 0] = 0

        # Cleaned layer band_2 HV (cHV)
        c_bnd2 = deep_clean(rw1)

        # Enhanced layer of cHH + cHV
        #enh_img = c_bnd1 + c_bnd2
        #avg = avg + np.mean(enh_img, axis=(0, 1))

        # Incident angle as a normalized layer
        inc_ang[:] = ((df.at[i, 'inc_angle'] - avg_inc - min_inc) / (max_inc - min_inc))

        # 5 layers containing enh and normalized raw with NO duplication of the data
        X1.append(np.dstack((rw1, c_bnd1)))
        X2.append(np.dstack((rw2, c_bnd2)))

    return np.array(X1), np.array(X2)

def dup_data(fet1, fet2, tar, num = 4):

    # confirm data length matches
    assert len(fet1) == len(tar) and len(fet2) == len(fet1)
    x1 = []
    x2 = []
    y = []
    if num == 1:
        return fet1, fet2, tar
    else:
        for i in range(len(fet1)):
            print('duplicated image #', i)
            # rotate images
            x1.extend(list(rotate_img(fet1[i], rotate=num)))
            x2.extend(list(rotate_img(fet2[i], rotate=num)))
            y.extend([tar[i]] * num)
        return np.array(x1), np.array(x2), np.array(y)

def bnd_Model():
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

    optimizer = Adam(lr= 0.002, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.01)
    #optimizer = Adam(lr=0.001, decay=0.0)
    #optimizer = keras.optimizers.SGD(lr=0.03, momentum=0.03, decay=0.01, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def getmodel():

    f_model = Sequential()
    #inp_bnd1 = Input(shape=input_shape)
    bnd1_model = bnd_Model()

    #inp_bnd2 = Input(shape=input_shape)
    bnd2_model = bnd_Model()

    #merged = Concatenate([bnd1_model, bnd2_model])
    #f_model = Model(inputs=[inp_bnd1, inp_bnd2], outputs=merged)
    f_model.add(Merge([bnd1_model, bnd2_model], mode='concat'))

    f_model.add(Dense(512, activation = 'relu'))
    f_model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.003, beta_1=0.99, beta_2=0.99, epsilon=1e-08, decay=0.01)
    # optimizer = Adam(lr=0.001, decay=0.0)
    # optimizer = keras.optimizers.SGD(lr=0.03, momentum=0.03, decay=0.01, nesterov=True)
    f_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return f_model

def import_training_data():
    # load the training data
    train = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/Train/train.json')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

    # option 1 fill the images with nan for inc_angle with mean of angles
    train['inc_angle'] = train['inc_angle'].fillna(train['inc_angle'].mean())
    #test['inc_angle'] = test['inc_angle'].fillna(test['inc_angle'].mean())

    # option 2 drop images with nan
    #train = train.dropna(axis=0, how='any')

    print('train data has been read')

def data_wrangling(order, flip):
    global train, X1_tr, X2_tr, y_r, X1_dup, X2_dup, y_dup

    path = '/Users/ohad/Google_Drive/DS_projects/K_Statoil/K_Statoil/data'

    X1_tr, X2_tr = prep_data(train, order=order)
    y_tr = np.array(train['is_iceberg'])

    # duplicating the data by rotating 90 degree 4 times:
    X1_dup, X2_dup, y_dup = dup_data(X1_tr, X2_tr, y_tr, num=flip)

def setup_input_shape(order, layer):
    global fet1_train, fet1_val, fet2_train, fet2_val, tar_train, tar_val, input_shape
    # input image dimensions
    img_rows, img_cols, layers = int(75 * order), int(75 * order), layer

    # configuring the input_shape and the x, y to the tensor flow structure
    if K.image_data_format() == 'channels_first':
        fet1_train = fet1_train.reshape(fet1_train.shape[0], layers, img_rows, img_cols)
        fet2_train = fet2_train.reshape(fet2_train.shape[0], layers, img_rows, img_cols)
        fet1_val = fet1_val.reshape(fet1_val.shape[0], layers, img_rows, img_cols)
        fet2_val = fet2_val.reshape(fet2_val.shape[0], layers, img_rows, img_cols)

        input_shape = (layers, img_rows, img_cols)
    else:
        fet1_train = fet1_train.reshape(fet1_train.shape[0], img_rows, img_cols, layers)
        fet2_train = fet2_train.reshape(fet2_train.shape[0], img_rows, img_cols, layers)
        fet1_val = fet1_val.reshape(fet1_val.shape[0], img_rows, img_cols, layers)
        fet2_val = fet2_val.reshape(fet2_val.shape[0], img_rows, img_cols, layers)
        input_shape = (img_rows, img_cols, layers)

def prep_submission():
    test = pd.read_json('/Users/ohad/Google_Drive/DS_projects/K_Statoil/test/test.json')
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

    # option2 drop images with nan
    test = test.dropna(axis=0, how='any')
    print('test has been read')

    X1_tst, X2_tst = prep_data(test, order)

    pred_test = model.predict([X1_tst, X2_tst])

    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

    scr = str(score[0]).replace(".", "")[0:5]

    submission.to_csv('submission_scr_{}.csv'.format(scr), index=False)

def prep_submission_FLOYD():
    test = pd.read_json('/data/Test/test.json')
    test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

    # option2 drop images with nan
    # test = test.dropna(axis=0, how='any')
    print('test has been read')

    X1_tst, X2_tst = prep_data(test, order)

    pred_test = model.predict([X1_tst, X2_tst])

    scr = str(score[0]).replace(".", "")[0:5]

    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

    submission.to_csv('/output/submission_scr_{}.csv'.format(scr), index=False)


#file_name = 'X_prep{}'.format(str(order).replace(".", ""))
#np.save(os.path.join(path, file_name), X_tr)
#file_name = 'y_prep{}'.format(str(order).replace(".", ""))
#np.save(os.path.join(path, file_name), y_tr)

#file_name = 'X_dup{}_ord{}_lyr{}'.format(flip, str(order).replace(".", ""), layers)
#np.save(os.path.join(path, file_name), X_dup)
#file_name = 'y_dup{}_ord{}_lyr{}'.format(flip, str(order).replace(".", ""), layers)
#np.save(os.path.join(path, file_name), y_dup)

# Order to enlarge our images
order = 1
# number of layer per image
layers = 2
# number of duplicate images
duplicate = 2

batch_size = 20
epochs = 50

import_training_data()

data_wrangling(order, duplicate)

# split train and validate data
fet1_train, fet1_val, fet2_train, fet2_val, tar_train, tar_val = my_tr_tst_split(X1_dup, X2_dup, y_dup,
                                                                                 split=0.25,
                                                                                 random_state=30)
setup_input_shape(order, layers)

# change train to float32 for better performance
fet1_train = fet1_train.astype('float32')
fet2_train = fet2_train.astype('float32')

fet1_val = fet1_val.astype('float32')
fet2_val = fet2_val.astype('float32')

print('fet1_train shape:', fet1_train.shape, 'fet2_train shape:', fet2_train.shape)
print(fet1_train.shape[0], 'train samples')
print(fet1_val.shape[0], 'test samples')


############################################################
###########  model setup  ##################################
############################################################

# Setup model parameters

earlyStopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


model = getmodel()
model.summary()

############################################################
###########  run model   ###################################
############################################################


model.fit([fet1_train, fet2_train], tar_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          validation_data=([fet1_val, fet2_val], tar_val))

model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate([fet1_val, fet2_val], tar_val, verbose=1)
print('Test val loss:', score[0])
print('Test val accuracy:', score[1])

prep_submission()

prep_submission_FLOYD()

for i in range(100):
    output = '{} of 100'.format(i)
    replace_cmd_line(output=output)
    time.sleep(0.05)