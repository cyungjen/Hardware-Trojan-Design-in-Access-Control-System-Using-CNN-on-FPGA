from keras.models import Model
import keras
# from keras.preprocessing import image as image_utils
import keras.utils as image
import numpy as np
np.set_printoptions(threshold=np.inf, suppress=True)
model = keras.models.load_model("model_n.h5")
# Create a new model to retrieve the outputs of each layer
intermediate_layer_model = Model(inputs=model.input,
                                  outputs=[layer.output for layer in model.layers])


def readimage(image_path):
    image_resize = image.load_img(image_path,color_mode = "grayscale", target_size=(46,46,1))
    image_resize_array = image.img_to_array(image_resize)
    image_resize_array_reshape = image_resize_array.reshape(1,46,46,1)
    path2 = 'pic_n.coe'
    f = open(path2, 'w')
    for i in range(28):
        for j in range(28):
            print(int(image_resize_array_reshape[0][i][j][0]),file=f)
    f.close
    return image_resize_array_reshape                                  

# Predict the outputs of each layer
img = readimage("data_2n/unknow/un (12).jpg")
intermediate_output = intermediate_layer_model.predict(img)

# Print the outputs of each layer
path = 'p_result.txt'

f = open(path, 'w')

for i, output in enumerate(intermediate_output):
    print(f'Output of layer {i}:',file=f)
    print(output.shape,file=f)
    # np.set_printoptions(threshold=sys.maxsize)
    print(output,file=f)

print(model.predict(img))