```python
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps.zip 
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps-test-set.zip 
```

    --2023-05-31 00:38:02--  https://storage.googleapis.com/learning-datasets/rps.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.200.128, 74.125.68.128, 74.125.24.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.200.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 200682221 (191M) [application/zip]
    Saving to: ‘rps.zip’
    
    rps.zip             100%[===================>] 191.38M  16.3MB/s    in 13s     
    
    2023-05-31 00:38:15 (14.7 MB/s) - ‘rps.zip’ saved [200682221/200682221]
    
    --2023-05-31 00:38:16--  https://storage.googleapis.com/learning-datasets/rps-test-set.zip
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.251.10.128, 142.250.4.128, 74.125.68.128, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.251.10.128|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 29516758 (28M) [application/zip]
    Saving to: ‘rps-test-set.zip’
    
    rps-test-set.zip    100%[===================>]  28.15M  12.6MB/s    in 2.2s    
    
    2023-05-31 00:38:18 (12.6 MB/s) - ‘rps-test-set.zip’ saved [29516758/29516758]
    



```python
!mv rps.zip ~/DataSet/ &&mv rps-test-set.zip ~/DataSet/
```


```python
import os
import zipfile

local_zip = '/home/codespace/DataSet/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('~/DataSet/')
zip_ref.close()

local_zip = '/home/codespace/DataSet/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('~/DataSet/')
zip_ref.close()
```


```python
rock_dir = os.path.join('~/DataSet/rps/rock')
paper_dir = os.path.join('~/DataSet/rps/paper')
scissors_dir = os.path.join('~/DataSet/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```

    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock04-047.png', 'rock07-k03-036.png', 'rock04-070.png', 'rock07-k03-000.png', 'rock02-116.png', 'rock04-107.png', 'rock03-060.png', 'rock03-058.png', 'rock05ck01-069.png', 'rock03-004.png']
    ['paper02-048.png', 'paper06-055.png', 'paper07-116.png', 'paper06-119.png', 'paper05-043.png', 'paper06-017.png', 'paper05-046.png', 'paper04-006.png', 'paper02-037.png', 'paper01-083.png']
    ['scissors03-008.png', 'testscissors01-113.png', 'scissors02-117.png', 'scissors03-061.png', 'testscissors03-045.png', 'scissors03-058.png', 'scissors02-074.png', 'testscissors02-008.png', 'scissors01-011.png', 'scissors04-031.png']



```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```


    
![png](exp5-2_files/exp5-2_4_0.png)
    



    
![png](exp5-2_files/exp5-2_4_1.png)
    



    
![png](exp5-2_files/exp5-2_4_2.png)
    



    
![png](exp5-2_files/exp5-2_4_3.png)
    



    
![png](exp5-2_files/exp5-2_4_4.png)
    



    
![png](exp5-2_files/exp5-2_4_5.png)
    



```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/home/codespace/DataSet/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "/home/codespace/DataSet/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")
```

    Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.


    2023-05-31 00:42:06.965079: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2023-05-31 00:42:06.965113: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2023-05-31 00:42:06.965139: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-b511a4): /proc/driver/nvidia/version does not exist
    2023-05-31 00:42:06.965708: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
     )                                                               
                                                                     
     conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
     2D)                                                             
                                                                     
     conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
     2D)                                                             
                                                                     
     conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
     2D)                                                             
                                                                     
     flatten (Flatten)           (None, 6272)              0         
                                                                     
     dropout (Dropout)           (None, 6272)              0         
                                                                     
     dense (Dense)               (None, 512)               3211776   
                                                                     
     dense_1 (Dense)             (None, 3)                 1539      
                                                                     
    =================================================================
    Total params: 3,473,475
    Trainable params: 3,473,475
    Non-trainable params: 0
    _________________________________________________________________


    2023-05-31 00:42:09.265797: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.


    Epoch 1/25


    2023-05-31 00:42:12.472458: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
    2023-05-31 00:42:12.520803: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
    2023-05-31 00:42:13.361096: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 176633856 exceeds 10% of free system memory.
    2023-05-31 00:42:13.473004: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 167215104 exceeds 10% of free system memory.


    20/20 [==============================] - 81s 4s/step - loss: 1.8150 - accuracy: 0.3552 - val_loss: 1.5109 - val_accuracy: 0.3333
    Epoch 2/25
    20/20 [==============================] - 74s 4s/step - loss: 1.1137 - accuracy: 0.3853 - val_loss: 1.0012 - val_accuracy: 0.4328
    Epoch 3/25
    20/20 [==============================] - 74s 4s/step - loss: 1.0897 - accuracy: 0.4528 - val_loss: 0.9516 - val_accuracy: 0.3522
    Epoch 4/25
    20/20 [==============================] - 74s 4s/step - loss: 1.0040 - accuracy: 0.4988 - val_loss: 0.8100 - val_accuracy: 0.5511
    Epoch 5/25
    20/20 [==============================] - 71s 4s/step - loss: 0.9036 - accuracy: 0.6083 - val_loss: 0.5554 - val_accuracy: 0.9113
    Epoch 6/25
    20/20 [==============================] - 72s 4s/step - loss: 0.7167 - accuracy: 0.6706 - val_loss: 0.3445 - val_accuracy: 0.9274
    Epoch 7/25
    20/20 [==============================] - 69s 3s/step - loss: 0.6951 - accuracy: 0.7036 - val_loss: 0.7432 - val_accuracy: 0.6210
    Epoch 8/25
    20/20 [==============================] - 69s 3s/step - loss: 0.5389 - accuracy: 0.7571 - val_loss: 0.1927 - val_accuracy: 0.9543
    Epoch 9/25
    20/20 [==============================] - 70s 3s/step - loss: 0.5342 - accuracy: 0.7996 - val_loss: 0.1901 - val_accuracy: 0.9839
    Epoch 10/25
    20/20 [==============================] - 69s 3s/step - loss: 0.3777 - accuracy: 0.8452 - val_loss: 0.2210 - val_accuracy: 0.9812
    Epoch 11/25
    20/20 [==============================] - 72s 4s/step - loss: 0.2833 - accuracy: 0.9016 - val_loss: 0.1520 - val_accuracy: 0.9704
    Epoch 12/25
    20/20 [==============================] - 72s 4s/step - loss: 0.3445 - accuracy: 0.8655 - val_loss: 0.1017 - val_accuracy: 0.9704
    Epoch 13/25
    20/20 [==============================] - 73s 4s/step - loss: 0.2396 - accuracy: 0.9048 - val_loss: 0.2184 - val_accuracy: 0.9167
    Epoch 14/25
    20/20 [==============================] - 71s 3s/step - loss: 0.1857 - accuracy: 0.9294 - val_loss: 0.0617 - val_accuracy: 0.9785
    Epoch 15/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1880 - accuracy: 0.9274 - val_loss: 0.0710 - val_accuracy: 0.9758
    Epoch 16/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1365 - accuracy: 0.9500 - val_loss: 0.0138 - val_accuracy: 1.0000
    Epoch 17/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1569 - accuracy: 0.9393 - val_loss: 0.5191 - val_accuracy: 0.7204
    Epoch 18/25
    20/20 [==============================] - 75s 4s/step - loss: 0.1663 - accuracy: 0.9313 - val_loss: 0.5619 - val_accuracy: 0.7554
    Epoch 19/25
    20/20 [==============================] - 77s 4s/step - loss: 0.1005 - accuracy: 0.9651 - val_loss: 0.3593 - val_accuracy: 0.8575
    Epoch 20/25
    20/20 [==============================] - 77s 4s/step - loss: 0.1248 - accuracy: 0.9532 - val_loss: 0.2169 - val_accuracy: 0.8978
    Epoch 21/25
    20/20 [==============================] - 75s 4s/step - loss: 0.0858 - accuracy: 0.9722 - val_loss: 0.1765 - val_accuracy: 0.9301
    Epoch 22/25
    20/20 [==============================] - 70s 3s/step - loss: 0.1096 - accuracy: 0.9571 - val_loss: 0.0787 - val_accuracy: 0.9731
    Epoch 23/25
    20/20 [==============================] - 71s 4s/step - loss: 0.1108 - accuracy: 0.9579 - val_loss: 0.0284 - val_accuracy: 0.9866
    Epoch 24/25
    20/20 [==============================] - 74s 4s/step - loss: 0.0746 - accuracy: 0.9710 - val_loss: 0.0194 - val_accuracy: 1.0000
    Epoch 25/25
    20/20 [==============================] - 72s 4s/step - loss: 0.0775 - accuracy: 0.9746 - val_loss: 0.0090 - val_accuracy: 1.0000



```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```


    
![png](exp5-2_files/exp5-2_6_0.png)
    



    <Figure size 640x480 with 0 Axes>

