# Part 6

## Task 1

- PCA
  
  - Epoch 30/30
    469/469 [==============================] - 2s 3ms/step - loss: 0.5936 - acc: 0.8098 - val_loss: 0.5801 - val_acc: 0.8148 
  
  - PCA MSE: x_train 0.0258 - x_test 0.0256

### Auto

#### Attempt 1

- Epoch 30/30
  469/469 [==============================] - 2s 4ms/step - loss: 0.5946 - acc: 0.8089 - val_loss: 0.5790 - val_acc: 0.8144

- val_loss: 0.0257 - val_mse: 0.0257

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(32*32))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 2

- val_loss: 0.1011 - val_mse: 0.1011

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(64, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64, activation = "relu"))
autoencoder.add(Dense(32*32,  activation = "softmax"))
autoencoder.add(Reshape((32,32,1)))
```

### Attempt 3

- val_loss: 0.0257 - val_mse: 0.0257

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(64))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64))
autoencoder.add(Dense(32*32))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 4

- val_loss: 0.1018 - val_mse: 0.1018

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(64, activation = "softmax"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64, activation = "softmax"))
autoencoder.add(Dense(32*32, activation = "softmax"))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 5

- val_loss: 0.0374 - val_mse: 0.0374

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(64, activation = "softmax"))
autoencoder.add(Dense(64))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64, activation = "softmax"))
autoencoder.add(Dense(64))
autoencoder.add(Dense(32*32, activation = "softmax"))
autoencoder.add(Dense(32*32))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 6

- val_loss: 0.0237 - val_mse: 0.0237

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(64, activation = "softmax"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64, activation = "softmax"))
autoencoder.add(Dense(32*32))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 7

- val_loss: 0.0116 - val_mse: 0.0116

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(128, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(128, activation = "relu"))
autoencoder.add(Dense(32*32, activation = "sigmoid"))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 8

- val_loss: 0.0110 - val_mse: 0.0110

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(128, activation = "relu"))
autoencoder.add(Dense(64, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(64, activation = "relu"))
autoencoder.add(Dense(128, activation = "relu"))
autoencoder.add(Dense(32*32, activation = "sigmoid"))
autoencoder.add(Reshape((32,32,1)))
```

#### Attempt 9

- val_loss: 0.0092 - val_mse: 0.0092

- Epoch 30/30
  469/469 [==============================] - 2s 4ms/step - loss: 0.2894 - acc: 0.9146 - val_loss: 0.2857 - val_acc: 0.9167

```python
autoencoder.add(Flatten(input_shape=(32,32,)))
autoencoder.add(Dense(256, activation = "relu"))
autoencoder.add(Dense(128, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
autoencoder.add(Dense(10, name='representation'))
autoencoder.add(Dense(128, activation = "relu"))
autoencoder.add(Dense(256, activation = "relu"))
autoencoder.add(Dense(32*32, activation = "sigmoid"))
autoencoder.add(Reshape((32,32,1)))
```

### Conv

#### Attempt 1

- val_loss: 0.0285 - val_mse: 0.0285

- Epoch 30/30
  469/469 [==============================] - 2s 4ms/step - loss: 0.5782 - acc: 0.8170 - val_loss: 0.5655 - val_acc: 0.8193

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(32, 3, strides=2, padding='same'))
conv_autoencoder.add(Flatten())
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(1*16*16))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(UpSampling2D((2, 2)))

conv_autoencoder.build((None,32,32,1))
```

#### Attempt 2

- val_loss: 0.0258 - val_mse: 0.025

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(64, 3, strides=2, padding='same'))
conv_autoencoder.add(Flatten())
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(1*16*16, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(UpSampling2D((2, 2)))

conv_autoencoder.build((None,32,32,1))
```

#### Attempt 3

- val_loss: 0.0243 - val_mse: 0.0243

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, 3, strides=2, padding='same', activation = "relu"))
conv_autoencoder.add(Conv2D(64, 3, strides=2, padding='same', activation = "relu"))
conv_autoencoder.add(Flatten())
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(1*16*16, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(UpSampling2D((2, 2)))

conv_autoencoder.build((None,32,32,1))
```

#### Attempt 4

- val_loss: 0.0221 - val_mse: 0.0221

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, 3, strides=2, padding='same', activation = "relu"))
conv_autoencoder.add(Conv2D(64, 3, strides=2, padding='same', activation = "relu"))
conv_autoencoder.add(Flatten())
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(64, activation = "relu"))
conv_autoencoder.add(Dense(1*16*16, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(UpSampling2D((2, 2)))

conv_autoencoder.build((None,32,32,1))
```

#### Attempt 5

- val_loss: 0.0098 - val_mse: 0.0098

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Flatten())
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(32*32*1))
conv_autoencoder.add(Reshape((32,32,1)))
conv_autoencoder.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
```

#### Attempt 6

- val_loss: 0.1028 - val_mse: 0.1028

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Flatten())
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(8*8*1, activation = "relu"))
conv_autoencoder.add(Reshape((8,8,1)))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
```

#### Attempt 7

- val_loss: 0.1028 - val_mse: 0.1028

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Flatten())
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
conv_autoencoder.add(Conv2D(1, (3, 3), padding="same"))
```

#### Attempt 8

- val_loss: 0.0078 - val_mse: 0.0078

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Flatten())
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(UpSampling2D((2, 2)))
#conv_autoencoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
conv_autoencoder.add(Conv2D(1, (3, 3), padding="same"))
```

#### Attempt 9

- val_loss: 3.3443e-06 - val_mse: 3.3443e-06

```python
inputs = Input((32,32,1))


conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#drop5 = Dropout(rate=0.5)(conv3)

flatten = Flatten()(conv3)
dense1 = Dense(8*8*1, activation = "relu")(flatten)
rep = Dense(10, name='representation')(dense1)
dense2 = Dense(8*8*1, activation = "relu")(rep)
dense2 = Dense(8*8*256, activation = "relu")(rep)
reshape = Reshape((8,8,256))(dense2)

## Now the decoder starts
up1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(reshape)
merge1 = concatenate([conv3,up1], axis = 3)
conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)

up2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
merge2 = concatenate([conv2,up2], axis = 3)
conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)

up3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
merge3 = concatenate([conv1,up3], axis = 3)
conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
conv7 = Conv2D(1,1, padding = 'same')(conv6)

model = Model(inputs = inputs, outputs = conv7)
model.summary()
conv_autoencoder = model
```

#### Attempt 10

- val_loss: 1.0966e-04 - val_mse: 1.0966e-04

```python
inputs = Input((32,32,1))


conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
#drop5 = Dropout(rate=0.5)(conv3)

flatten = Flatten()(conv3)
dense1 = Dense(8*8*1, activation = "relu")(flatten)
rep = Dense(10, name='representation')(dense1)
dense2 = Dense(8*8*1, activation = "relu")(rep)
dense2 = Dense(8*8*256)(rep)
reshape = Reshape((8,8,256))(dense2)

## Now the decoder starts
up1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(reshape)
merge1 = concatenate([conv3,up1], axis = 3)
conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)

up2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
merge2 = concatenate([conv2,up2], axis = 3)
conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)

up3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up3)
conv7 = Conv2D(1,1, padding = 'same')(conv6)

model = Model(inputs = inputs, outputs = conv7)
model.summary()
conv_autoencoder = model
```

#### Attempt 11

```python
conv_autoencoder = Sequential()
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(MaxPooling2D((2, 2), padding="same"))
conv_autoencoder.add(Flatten())
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
conv_autoencoder.add(Dense(64, activation = "relu"))
# The representation has dimensionality 10, do not change the dimensionality
conv_autoencoder.add(Dense(10, name='representation'))
conv_autoencoder.add(Dense(64, activation = "relu"))
conv_autoencoder.add(Dense(16*16*1, activation = "relu"))
conv_autoencoder.add(Reshape((16,16,1)))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
#conv_autoencoder.add(UpSampling2D((2, 2)))
conv_autoencoder.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
conv_autoencoder.add(UpSampling2D((2, 2)))
#conv_autoencoder.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))
conv_autoencoder.add(Conv2D(1, (3, 3), padding="same"))
```

- Epoch 30/30
  469/469 [==============================] - 2s 4ms/step - loss: 0.2337 - acc: 0.9345 - val_loss: 0.2336 - val_acc: 0.9342

- val_loss: 0.0081 - val_mse: 0.0081

## Task 2

- MSE with given loss function: MSE (trained with Custom loss):  0.006218516267836094

### Lhdr

https://arxiv.org/pdf/1803.04189.pdfhttps://arxiv.org/pdf/1803.04189.pdf

```python
def custom_loss(true_values, predicted_values):
  
  #     ...    
  #     Define here your loss
  #     ...
  #(fθ (ˆx) − ˆy)2/(fθ (ˆx) + 0.01)2
  return np.pow(true_values - predicted_values, 2) / np.pow((true_values + 0.01),2)
  #return []
```

MSE (trained with Custom loss):  0.007172251585870981

### SSIM

[Structural similarity - Wikipedia](https://en.wikipedia.org/wiki/Structural_similarity)

```python
return 1 - tf.reduce_mean(tf.image.ssim(true_values, predicted_values, 2.0))
```

MSE (trained with Custom loss):  0.0056237876415252686

MSE of 1.2691 when dividing instead of subtracting

MSE (trained with Custom loss):  0.014981732703745365 when using 255 isnteasd of 2

### MS-SSIM

MSE (trained with Custom loss):  0.005623974371701479

```python
return 1- tf.reduce_mean(tf.image.ssim_multiscale(tf.expand_dims(true_values,axis=0), tf.expand_dims(predicted_values,axis=0), max_val = 255, filter_size = 1))
```

#### PSNR

MSE (trained with Custom loss):  0.0074236588552594185

```python
return 1 / tf.reduce_mean(tf.image.psnr(true_values, predicted_values, 1, name=None))
```


