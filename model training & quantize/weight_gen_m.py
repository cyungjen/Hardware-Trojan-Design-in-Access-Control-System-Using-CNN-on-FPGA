import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D, MaxPooling2D
from keras.optimizers import Adam
import keras.utils as image
from keras.utils import to_categorical
import tensorflow.python.keras.callbacks as tf_callback
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten,Reshape
# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.utils import shuffle
import sys
import numpy as np
from numpy import empty
import math
np.set_printoptions(suppress=True)
df = pd.read_csv("data_2m.csv")
dfd = df.drop([0],axis=0)
data = dfd.sample(frac=1)


def readimage(image_path):
    image_resize = image.load_img(image_path,color_mode = "grayscale", target_size=(46,46,1))
    image_resize_array = image.img_to_array(image_resize)
    image_resize_array_reshape = image_resize_array.reshape(46,46,1)

    return image_resize_array_reshape


def f_quantize(x):
   
    # decimal
    b_num = 12
    # integer 
    a_num = 4
    # whole num
    num = a_num+b_num

    int_val = 0
    del_val = 0
    del_vec = [0,0,0,0,0,0,0,0,0,0,0,0]#decimal part

    if (x>0):
        x_fix = x
        if (x>7.999755859375):
            x_fix = 7.999755859375
        
        int_val = math.floor(x_fix)
        del_val = x_fix - int_val

        # get int part in binary 
        int_vec = list(format(int_val, "b").zfill(4))
        for i in range(4):
            int_vec[i] = int(int_vec[i]) 
        # get dec part in binary
        tmp = del_val
        for i in range(b_num):
            w_bin = 2**-(i+1)
            if(tmp >= w_bin):
                del_vec[i]=1
                tmp = tmp-w_bin
            else:
                del_vec[i]=0
        num_vec = int_vec+del_vec
    # x<0   
    else:
        x_fix=abs(x)
        if x_fix > 7.999755859375:
            x_fix = 7.999755859375

        int_val = math.floor(x_fix)
        del_val = x_fix - int_val

        # get int part in binary 
        int_vec = list(format(int_val, "b").zfill(4))
        for i in range(4):
            int_vec[i] = int(int_vec[i]) 
        # get dec part in binary
        tmp = del_val
        for i in range(b_num):
            w_bin = 2**-(i+1)
            if(tmp >= w_bin):
                del_vec[i]=1
                tmp = tmp-w_bin
            else:
                del_vec[i]=0
        num_vec = int_vec+del_vec
        # 取反
        for i in range(16):
            if num_vec[i]==1:
                num_vec[i]=0
            else:
                num_vec[i]=1
    encode_out = 0
    for i in range(16):
        w_dec=2**(15-i)
        if(num_vec[i]==1):
            encode_out = encode_out + w_dec
    if(x<0 and encode_out<65535):
        encode_out = encode_out+1
    
    return str(encode_out)

def f_quantize_conv2(x):
    # decimal
    b_num = 12
    
    del_val = 0
    del_vec = [0,0,0,0,0,0,0,0,0,0,0,0]#decimal part

    if (x>0):
        x_fix = x
        if (x>7.999755859375):
            x_fix = 7.999755859375
        
        del_val = x_fix 

        # get int part in binary 
        # int_vec = list(format(int_val, "b").zfill(4))
        # for i in range(4):
        #     int_vec[i] = int(int_vec[i]) 
        # get dec part in binary
        tmp = del_val
        for i in range(b_num):
            w_bin = 2**-(i+1)
            if(tmp >= w_bin):
                del_vec[i]=1
                tmp = tmp-w_bin
            else:
                del_vec[i]=0
        num_vec = del_vec
    # x<0   
    else:
        x_fix=abs(x)
        if x_fix > 7.999755859375:
            x_fix = 7.999755859375

        del_val = x_fix

        # get int part in binary 
        # int_vec = list(format(int_val, "b").zfill(4))
        # for i in range(4):
        #     int_vec[i] = int(int_vec[i]) 
        # get dec part in binary
        tmp = del_val
        for i in range(b_num):
            w_bin = 2**-(i+1)
            if(tmp >= w_bin):
                del_vec[i]=1
                tmp = tmp-w_bin
            else:
                del_vec[i]=0
        num_vec = del_vec
        # 取反
        for i in range(12):
            if num_vec[i]==1:
                num_vec[i]=0
            else:
                num_vec[i]=1
    encode_out = 0
    for i in range(12):
        w_dec=2**(11-i)
        if(num_vec[i]==1):
            encode_out = encode_out + w_dec
    if(x<0):
        encode_out = encode_out+1

    encode_out = format(encode_out,'012b')
    return str(encode_out)





X = []
Y = []
num_categories = 3

for i in range(len(data)):
    img_path = f"data_2m/{(data.iloc[i][1])}/{(data.iloc[i][2])}"
    X.append(readimage(img_path))
    Y.append(int(data.iloc[i][4]))

#轉np array
X_np = np.array(X)
X_normal = X_np#不做正規化
Y_np = np.array(Y)

#label做one hot encoding
y_category = to_categorical(Y_np, num_categories)
# y_category = keras.utils.to_categorical(Y_np)
#plt.imshow(X_np[2],cmap='gray')

# 測試看看 y_train_category[0]

###step2:建立CNN模型###
# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(46, 46, 3),padding='valid'),
#     MaxPooling2D((2, 2)),
#     Conv2D(8, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(6, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_normal, y_category, batch_size=32, epochs=10, validation_split=0.2,verbose=1)

model = Sequential()
# 加入第一層卷積層
model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=(46,46,1), activation='relu', padding='valid'))
# 加入第一層池化層
model.add(MaxPooling2D(pool_size=(2,2)))
# 加入第二層卷積層
model.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='valid'))
# 加入第二層池化層
model.add(MaxPooling2D(pool_size=(2,2)))
# 加入平坦層
model.add(Flatten())
# 加入隱藏層
# model.add(Dense(32, activation='relu'))
# 加入隱藏層
model.add(Dense(64, activation='relu'))
# 加入輸出層
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # Assuming X_train and y_train are the training data and labels
# model.fit(X_train, y_train, epochs=10, batch_size=64)
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_normal, y_category, epochs=60,batch_size=32, verbose=1,
             validation_split=0.2)






# weight,biases = model.layers[0].get_weights()
# print(w)
model.save('model_2m.h5')

for layer in model.layers:
    for weight in layer.weights:
        print (weight.name,weight.shape)
c=0
for layer in model.layers: 
    c = c+1
    if c == 6:
        str_bias1 = ""
        list_dense1 = []
        get_dense1 = layer.get_weights()[0]
        get_bias1 = layer.get_weights()[1]
        print(get_dense1)
        for j in range(64):
            for i in range(800):
                list_dense1.append(get_dense1[i][j])
        for i in range(64):
            str_bias1 += f_quantize(get_bias1[i])
            str_bias1 += ",\n"
        #write coe file
        with open('weight_2m/dense1m.coe', 'w') as f:
            print('memory_initialization_radix = 10;\nmemory_initialization_vector =',file=f)
            np.set_printoptions(threshold=sys.maxsize)
            for j in range(51200):
                print( f_quantize(list_dense1[j]),file=f,end="")
                if j!=51199:
                    print(",",file=f,)
                else:
                    print(";",file=f,end="")
        with open('weight_2m/bias1m.coe', 'w') as f:
            np.set_printoptions(threshold=sys.maxsize)
            print('memory_initialization_radix = 10;\nmemory_initialization_vector =',file=f)
            print( str_bias1, file=f)
    if c == 7:
        str_dense2 = ""
        str_bias2 = ""
        get_dense2 = layer.get_weights()[0]
        get_bias2 = layer.get_weights()[1]
        print(get_bias2)
        for j in range(3):
            for i in range(64):
                str_dense2 += f_quantize(get_dense2[i][j])
                str_dense2 += ",\n"
        for i in range(3):
            str_bias2 += f_quantize(get_bias2[i])
            str_bias2 += ","
        with open('weight_2m/dense2m.coe', 'w') as f:
            print('memory_initialization_radix = 10;\nmemory_initialization_vector =',file=f)
            np.set_printoptions(threshold=sys.maxsize)
            print( str_dense2, file=f)
        with open('weight_2m/bias2m.coe', 'w') as f:
            print('memory_initialization_radix = 10;\nmemory_initialization_vector =',file=f)
            np.set_printoptions(threshold=sys.maxsize)
            print( str_bias2, file=f )
label= 'conv'
kr = 'KRN_ROWS'
kc = 'KRN_COLS'
f = 'FILTERS'
cont_conv = 0
for idx,i in enumerate(model.layers):
    if(isinstance(i, Conv2D)):
        if(cont_conv==0):
            res = ""
            w = i.get_weights()[0]
            print("w=",w.shape)
            # Conversion of weights array.
            new_w = empty(shape=(w.shape[3],w.shape[0],w.shape[1]))
            print("new shape=",new_w.shape)
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    for filter in range(w.shape[3]):
                        new_w[filter][row][col] = w[row][col][0][filter]
            w = new_w

            # Weights: (label)_weights[kr][kc][f].
            for filter in range(w.shape[0]):
                res = 'memory_initialization_radix = 10;\nmemory_initialization_vector = \n'
                for row in range(w.shape[1]-1,-1,-1):
                    for col in range(w.shape[2]-1,-1,-1):
                        res += f_quantize(float(w[filter][row][col]))
                        if (col != 0):
                            res += ', '
                    if (row !=0):
                        res += ', '
                res += ';\n'
                with open('weight_2m/c1_w_'+str(filter+1)+'m.coe', 'w') as f:
                    # print('writing \'conv_weights.coe\' file... ', end='')
                    # print('/*\n * This file is auto-generated by gen-weights.py\n */\n',file=f)
                    # print('#pragma once\n\n#include "definitions.h"\n\n',file=f)
                    arrays_def_str = res
                    print(arrays_def_str, file=f)
                    # print('done.')
            cont_conv = cont_conv+1

        # sec_conv layer     
        elif(cont_conv==1):
            w2 = i.get_weights()[0]
            print("w2=",w2.shape)

            for row in range(w2.shape[0]):
                res=''
                res = 'memory_initialization_radix = 2;\nmemory_initialization_vector = \n'
                for filter in range(w2.shape[3]):
                    for depth in range(w2.shape[2]):
                        for col in range(w2.shape[1]):
                            res += f_quantize_conv2(float(w2[row][col][depth][filter]))
                    if (filter != w2.shape[3]-1):
                            res += ',\n'
                res += ';'
                    
                with open('weight_2m/c2_w_'+str(row+1)+'m.coe', 'w') as f:
                    # print('writing \'conv_weights.coe\' file... ', end='')
                    # print('/*\n * This file is auto-generated by gen-weights.py\n */\n',file=f)
                    # print('#pragma once\n\n#include "definitions.h"\n\n',file=f)
                    arrays_def_str = res
                    print(arrays_def_str, file=f)
                    # print('done.')
            
            cont_conv = cont_conv+1
            ########################decimal##########################
            res = " "
            w2 = i.get_weights()[0]
            print("w=",w2.shape)
            # Conversion of weights array.
            new_w = empty(shape=(8,16,3,3))
            print("new shape=",new_w.shape)
            
            for row in range(w2.shape[0]):
                for col in range(w2.shape[1]):
                    for depth in range(w2.shape[2]):
                        for filter in range(w2.shape[3]):
                            new_w[filter][depth][row][col] = w2[row][col][depth][filter]
            w2 = new_w

            # Weights: (label)_weights[kr][kc][f].
            res += '// ' + label.capitalize() + ' layer weights.\n'
            res += 'float ' + 'conv' + '_weights [' + 'FILTERS' + '][' + 'DEPTH' + '][' + 'KRN_ROWS' + '][' \
                    + 'KRN_COLS' + ']\n\t= {\n'
            for filter in range(w2.shape[0]):
                res += '\t\t\t{\n'
                for depth in range(w2.shape[1]):
                    res += '\t\t\t{\n'
                    for row in range(w2.shape[2]):
                        res += '\t\t\t\t{ '
                        for col in range(w2.shape[3]):
                            res += str(float(w2[filter][depth][row][col]))
                            if (col != w2.shape[3]-1):
                                res += ', '
                        res += ' }'
                        if(row != w2.shape[2] -1):
                            res += ','
                        res += '\n'
                    res += '\t\t\t}'
                    if(depth != w2.shape[1] -1):
                        res += ','
                    res += '\n'
                res += '\t\t\t}'
                if(filter != w2.shape[0] -1):
                    res += ','
                res += '\n'
            res +='\t\t};\n\n'

            with open('conv22d_weights.h', 'w') as f:
                print('writing \'conv_weights2d.h\' file... ', end='')
                print('/*\n * This file is auto-generated by gen-weights.py\n */\n',file=f)
                print('#pragma once\n\n#include "definitions.h"\n\n',file=f)
                arrays_def_str = res
                print(arrays_def_str, file=f)
                print('done.')





# label =  'dense'
# size0 = 'FLAT_SIZE'
# size1 = 'DENSE_SIZE'
# cont_dense = 0

# for idx,i in enumerate(model.layers):
#     if(isinstance(i, Dense)):
#         if(cont_dense==0):
#             res = " "
#             w = i.get_weights()[0]
            
#             res = ''

#             # Conversion of weights array.
#             pool_img_r = 10
#             pool_img_c = 10
#             conv_filter_num = 8
#             dense_size = 64
#             tmp = empty(shape=(pool_img_r, pool_img_c, conv_filter_num, dense_size))
#             print(tmp.shape)
#             index2 = 0
#             for i in range(0,pool_img_r):
#                 for j in range(0,pool_img_c):
#                     for f in range(0,conv_filter_num):
#                         for d in range(0,dense_size):
#                             tmp[i][j][f][d] = w[index2][d]
#                         index2 += 1
#             print(tmp.shape)
#             index = 0
#             new_w2 = empty(w.shape)
#             print(w.shape)
#             for f in range(conv_filter_num):
#                 for i in range(pool_img_r):
#                     for j in range(pool_img_c):
#                         for d in range(dense_size):
#                             new_w2[index][d] = tmp[i][j][f][d]
#                         index += 1
#             w = new_w2

#             # Weights: (label)_weights[size0][size1].
#             res += '// ' + 'DENSE' + ' layer weights.\n'
#             res += 'float ' + 'dense' + '_weights[' + 'FLAT_SIZE' + '][' + 'DENSE_SIZE' + ']\n\t = {\n'
#             for i in range(w.shape[0]):
#                 res += '\t\t\t{ '
#                 for j in range(w.shape[1]):
#                     res += str(float(w[i][j]))
#                     if j != w.shape[1] - 1:
#                         res += ', '
#                 res += ' }'
#                 if i != w.shape[0] - 1:
#                     res += ','
#                 res += '\n'
#             res += '\t\t};\n\n'

#             with open('dense_weights.h', 'w') as f:
#                 print('writing \'dense_weights.h\' file... ', end='')
#                 print('/*\n * This file is auto-generated by gen-weights.py\n */\n',file=f)
#                 print('#pragma once\n\n#include "definitions.h"\n\n', file=f)
#                 arrays_def_str = res
#                 print(arrays_def_str, file=f)
#                 print('done.')
#             cont_dense=cont_dense+1
#         # dense_1
#         elif(cont_dense==1):
#             res = " "
#             w = i.get_weights()[0]
            
#             res = ''

#             # Conversion of weights array.
            

#             # Weights: (label)_weights[size0][size1].
#             res += '// ' + 'DENSE' + ' layer weights.\n'
#             res += 'float ' + 'dense1' + '_weights[' + 'FLAT_SIZE' + '][' + 'DENSE_SIZE' + ']\n\t = {\n'
#             for i in range(w.shape[0]):
#                 res += '\t\t\t{ '
#                 for j in range(w.shape[1]):
#                     res += str(float(w[i][j]))
#                     if j != w.shape[1] - 1:
#                         res += ', '
#                 res += ' }'
#                 if i != w.shape[0] - 1:
#                     res += ','
#                 res += '\n'
#             res += '\t\t};\n\n'

#             with open('dense1_weights.h', 'w') as f:
#                 print('writing \'dense1_weights.h\' file... ', end='')
#                 print('/*\n * This file is auto-generated by gen-weights.py\n */\n',file=f)
#                 print('#pragma once\n\n#include "definitions.h"\n\n', file=f)
#                 arrays_def_str = res
#                 print(arrays_def_str, file=f)
#                 print('done.')


