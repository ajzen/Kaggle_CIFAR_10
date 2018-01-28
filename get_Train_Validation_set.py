import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import misc
import random
import keras



def get_train_set():

    i = 0
    train_num_images = 0
    images = []
    y = []
    Y_dict = {}
    for dirnames in os.listdir("data/train_data"):
#         print(dirnames)
        Y_dict[dirnames] = i
        for filename in os.listdir("data/train_data/"+ dirnames):
            train_num_images = train_num_images + 1
            #print(filenames,"\n")
            # img = mpimg.imread("data/train_valid/"+ dirnames + "/" + filename) or  img =  misc.imread("data/train_valid/"+ dirnames + "/" + filename)
            img =  misc.imread("data/train_data/"+ dirnames + "/" + filename)
    #         plt.imshow(img)
    #         plt.show()
            img = misc.imresize(img , (224,224,3) )
            if img is not None:
                images.append(img)
                y.append(i)
    #             print(img)
    #             print(y)

        i = i + 1
    
    c = list(zip(images, y))

    random.shuffle(c)

    X,Y = zip(*c)
    
    train_set = np.reshape(X , (train_num_images,224,224,3) )
    train_out = keras.utils.to_categorical(Y,num_classes=10)
    print(train_set.shape)
    return (train_set , train_out)



# FOr valid_data

def get_valid_data():
    i = 0
    valid_num_images = 0
    v_images = []
    v_y = []
    v_Y_dict = {}
    for v_dirnames in os.listdir("data/valid_data"):
#         print(v_dirnames)
        v_Y_dict[v_dirnames] = i
        for v_filename in os.listdir("data/valid_data/"+ v_dirnames):
            valid_num_images = valid_num_images + 1
            #print(filenames,"\n")
            # img = mpimg.imread("data/train_valid/"+ dirnames + "/" + filename) or  img =  misc.imread("data/train_valid/"+ dirnames + "/" + filename)
            v_img =  misc.imread("data/valid_data/"+ v_dirnames + "/" + v_filename)
    #         plt.imshow(v_img)
    #         plt.show()
            v_img = misc.imresize(v_img , (224,224,3) )
            if v_img is not None:
                v_images.append(v_img)
                v_y.append(i)
    #             print(v_img)
    #             print(v_y)

        i = i + 1
        
    v_c = list(zip(v_images, v_y))

    random.shuffle(v_c)

    v_X,v_Y = zip(*v_c)
    
    validation_set = np.reshape(v_X , (valid_num_images,224,224,3) )
    validation_out = keras.utils.to_categorical(v_Y,num_classes=10)
    
    return (validation_set,validation_out)
        
