from segmentation_models import Unet
import segmentation_models as sm
from segmentation_models.metrics import iou_score
from segmentation_models.losses import bce_jaccard_loss
from keras import backend as K
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import os
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def sortKey(x):
    x = re.findall(r'\d+', x)
    return int(x[0])

def readimg(path, img_list):
    file_list = os.listdir(path)
    if any(x is None for x in file_list):
        print("no pictures")
    else:
        print("picture exist")
        count = 0
        for filename in sorted(file_list, key = sortKey): 
            # if count == 100:
            #     break
            temp_name = "{0}/{1}".format(path, filename)
            img = image.load_img(
                temp_name, target_size=(32, 32))
            img = image.img_to_array(img)
            img_list.append(img)
            count += 1

def plot_training_history(history):
    """
    Plots model training history 
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou_score"], label="Train iou")
    ax_acc.plot(history.epoch, history.history["val_iou_score"], label="Validation iou")
    ax_acc.legend()
    fig.savefig("result.jpg")

def main():
    DEFECT_PATH = "../A/defect"
    DEFECT_MASK_PATH = "../A/defect_mask"
    MASTER_PATH = "../A/master"
    MASTER_MASK_PATH = "../A/master_mask"

    X = []
    y = []
    y_master = []
    #read images
    readimg(DEFECT_PATH, X)
    readimg(MASTER_PATH, X)
    readimg(DEFECT_MASK_PATH, y)
    readimg(MASTER_MASK_PATH, y_master)

    #populate mask for master
    y_master = y_master * 141
    y = y + y_master
    X = np.asarray(X)
    y = np.asarray(y)
    print(X.shape, y.shape)
    X = X / 255.0
    y = y / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)

    figz = plt.figure(figsize=(8,8))
    figz.add_subplot(1,1,1)
    plt.imshow(X_train[5])
    figz.add_subplot(1,2,2)
    plt.imshow(y_train[5])
    plt.show()
    # reduces learning rate on plateau
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown= 10,
                               patience=10,verbose =1,
                               min_lr=0.1e-5)
    mode_autosave = ModelCheckpoint("./weights/vgg16imgsize.h5",monitor='val_iou_score', 
                                    mode = 'max', save_best_only=True, verbose=1, period =10)
    # stop learining as metric on validatopn stop increasing
    early_stopping = EarlyStopping(patience=10, verbose=1, mode = 'auto') 
    callbacks = [mode_autosave, lr_reducer]

    BACKBONE = 'efficientnetb0'
    preprocess = sm.get_preprocessing(BACKBONE)
    X_train = preprocess(X_train)
    X_val = preprocess(X_test)
    
    model = Unet(BACKBONE, input_shape= (32, 32, 3), classes = 1, encoder_weights='imagenet', activation='sigmoid')
    print(model.summary())
    model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])
    print('start fit')
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=15,
        epochs=30,
        validation_data=(X_test, y_test),
        verbose=2,
        callbacks = callbacks
    )
    plot_training_history(history)
    model.save('UNET.h5')

if __name__ == "__main__":
    main()