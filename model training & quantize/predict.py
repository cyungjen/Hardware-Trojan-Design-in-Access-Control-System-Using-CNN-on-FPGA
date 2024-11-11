import tensorflow as tf
import numpy as np
from PIL import Image
# import cv2
# 加载模型
model = tf.keras.models.load_model('model_2n.h5')

# 加载预测图像
image = Image.open('r1.jpg')
image = image.convert('L')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.resize((28, 28 )) # 调整图像大小
image_array = np.array(image)
# image_array = image_array / 255.0 # 归一化

# 添加批次维度并进行预测
image_tensor = np.expand_dims(image_array, axis=0)
predictions = model.predict(image_tensor)

# 打印预测结果
print(predictions)