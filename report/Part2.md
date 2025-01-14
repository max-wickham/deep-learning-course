# Part 2

## Task 1

- Activation function
- Pooling
- Convolutions
- Parallel convolutions

### Notes

- Doing decreasing the number of convolutions between layers is better than increasing.
- Large number of convolutions increases score up to a limit.
- Adding some dense layers also increases accuracy until hitting over fitting maximum. 
- Parallel convolutions gave a bad accuracy with very long training time.

### Results

#### Test 1:

Epoch 20/20 1563/1563 [==============================] - 6s 4ms/step - loss: 1.2030 - accuracy: 0.5816 Validation loss: 1.2316844463348389 Validation accuracy: 0.5673999786376953

```python
model = Sequential()
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 2:

Epoch 20/20 1563/1563 [==============================] - 7s 4ms/step - loss: 1.0469 - accuracy: 0.6384 Validation loss: 1.1108617782592773 Validation accuracy: 0.6105999946594238

```Python
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 3:

Epoch 20/20 1563/1563 [==============================] - 7s 4ms/step - loss: 0.9130 - accuracy: 0.6870 Validation loss: 1.1089383363723755 Validation accuracy: 0.6190999746322632

```python
model = Sequential()
model.add(Conv2D(32, (5,5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 4:

Epoch 20/20 1563/1563 [==============================] - 8s 5ms/step - loss: 0.8332 - accuracy: 0.7114 Validation loss: 0.9513970017433167 Validation accuracy: 0.6657000184059143

```python
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (5,5), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 5:

Epoch 20/20 1563/1563 [==============================] - 7s 5ms/step - loss: 1.0382 - accuracy: 0.6407 Validation loss: 1.1007344722747803 Validation accuracy: 0.6114000082015991

```python
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 6:

Epoch 20/20 1563/1563 [==============================] - 7s 5ms/step - loss: 1.0249 - accuracy: 0.6407 Validation loss: 1.0473380088806152 Validation accuracy: 0.6340000033378601

```python
model = Sequential()
model.add(Conv2D(64, (5,5), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 7:

Epoch 20/20 1563/1563 [==============================] - 11s 7ms/step - loss: 0.8416 - accuracy: 0.7072 Validation loss: 0.9306763410568237 Validation accuracy: 0.6777999997138977

```python
model = Sequential()
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 8:

Epoch 20/20 1563/1563 [==============================] - 12s 7ms/step - loss: 0.7529 - accuracy: 0.7366 Validation loss: 0.9446598887443542 Validation accuracy: 0.6764000058174133

```python
model = Sequential()
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(16, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 9:

Epoch 20/20 1563/1563 [==============================] - 16s 10ms/step - loss: 0.5315 - accuracy: 0.8172 Validation loss: 0.856573224067688 Validation accuracy: 0.7214999794960022 Time: 5:31

```python
model = Sequential()
model.add(Conv2D(128, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='reluq
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 9.5

1563/1563 [==============================] - 58s 37ms/step - loss: 0.5517 - accuracy: 0.8087 Validation loss: 0.8764972686767578 Validation accuracy: 0.7009999752044678

```python
model = Sequential()
model.add(Conv2D(16, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 10:

Epoch 20/20 1563/1563 [==============================] - 24s 16ms/step - loss: 0.4904 - accuracy: 0.8304 Validation loss: 0.8812670111656189 Validation accuracy: 0.7225000262260437 Time 8:13

```python
model = Sequential()
model.add(Conv2D(256, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### Test 11:

Epoch 20/20 1563/1563 [==============================] - 12s 8ms/step - loss: 0.7472 - accuracy: 0.7484 Validation loss: 1.0681636333465576 Validation accuracy: 0.64410001039505 Time: 5:42

```python
input_shape = Input(shape=x_train.shape[1:])

tower_1 = Conv2D(64, (3,3), padding='same', input_shape=x_train.shape[1:])(input_shape)
#tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_2 = Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:])(input_shape)
#tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)
tower_1 = Activation('relu')(tower_2)
merged = concatenate([tower_1, tower_2], axis=1)
merged = Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:], activation = 'relu')(merged)
merged = Flatten()(merged)

out = Dense(10, activation='softmax')(merged)
model = Model(input_shape, out)
```

#### Test 12:

```python
tower_1 = Conv2D(64, (3,3), padding='same', input_shape=x_train.shape[1:])(input_shape)
tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_2 = Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:])(input_shape)
tower_2 = Conv2D(32, (3,3), padding='same')(tower_2)
#tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)
tower_1 = Activation('relu')(tower_2)
merged = concatenate([tower_1, tower_2], axis=1)
merged = Flatten()(merged)

out = Dense(10, activation='softmax')(merged)
model = Model(input_shape, out)
model.summary()
```

Epoch 20/20 1563/1563 [==============================] - 10s 6ms/step - loss: 0.7265 - accuracy: 0.7549 Validation loss: 1.0539156198501587 Validation accuracy: 0.6521000266075134

#### Test 13:

```python
input_shape = Input(shape=x_train.shape[1:])

tower_1 = Conv2D(64, (3,3), padding='same', input_shape=x_train.shape[1:])(input_shape)
tower_1 = MaxPooling2D((2, 2), padding='same')(tower_1)
tower_1 = Activation('relu')(tower_1)
tower_2 = Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:])(input_shape)
tower_2 = Activation('relu')(tower_2)
tower_2 = Conv2D(32, (3,3), padding='same')(tower_2)
#tower_2 = MaxPooling2D((2, 2), padding='same')(tower_2)
tower_2 = Activation('relu')(tower_2)
merged = concatenate([tower_1, tower_2], axis=1)
merged = Flatten()(merged)
out = Dense(15, activation='relu')(merged)
out = Dense(10, activation='softmax')(out)
model = Model(input_shape, out)
```

## Task 2

### Notes

### Results

#### Attempt 1:

14/14 [==============================] - 3s 208ms/step - loss: 14.1812 - mean_absolute_percentage_error: 14.1812 - val_loss: 70.4299 - val_mean_absolute_percentage_error: 70.4299 Epoch 99/100 14/14 [==============================] - 3s 210ms/step - loss: 13.9132 - mean_absolute_percentage_error: 13.9132 - val_loss: 65.4726 - val_mean_absolute_percentage_error: 65.4726 Epoch 100/100 14/14 [==============================] - 3s 210ms/step - loss: 13.9263 - mean_absolute_percentage_error: 13.9263 - val_loss: 78.4035 - val_mean_absolute_percentage_error: 78.4035

```python
model = Sequential()
model.add(Conv2D(16, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
```

- Probably over-fitting

- Try reduce dense layers

- Add larger convolution size

#### Attempt 2:

14/14 [==============================] - 2s 133ms/step - loss: 11.9485 - mean_absolute_percentage_error: 11.9485 - val_loss: 56.4226 - val_mean_absolute_percentage_error: 56.4226 Epoch 99/100 14/14 [==============================] - 2s 135ms/step - loss: 11.0558 - mean_absolute_percentage_error: 11.0558 - val_loss: 67.4681 - val_mean_absolute_percentage_error: 67.4681 Epoch 100/100 14/14 [==============================] - 2s 131ms/step - loss: 11.0771 - mean_absolute_percentage_error: 11.0771 - val_loss: 61.0997 - val_mean_absolute_percentage_error: 61.0997

```python
model = Sequential()
model.add(Conv2D(16, (8,8), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
```

- Better generalisation and better validation accuracy

- Try reducing model complexity even further

#### Attempt 3:

14/14 [==============================] - 1s 80ms/step - loss: 14.1765 - mean_absolute_percentage_error: 14.1765 - val_loss: 72.6673 - val_mean_absolute_percentage_error: 72.6673 Epoch 99/100 14/14 [==============================] - 1s 81ms/step - loss: 14.0107 - mean_absolute_percentage_error: 14.0107 - val_loss: 67.9697 - val_mean_absolute_percentage_error: 67.9697 Epoch 100/100 14/14 [==============================] - 1s 85ms/step - loss: 13.1558 - mean_absolute_percentage_error: 13.1558 - val_loss: 79.3290 - val_mean_absolute_percentage_error: 79.3289

```python
model = Sequential()
model.add(Conv2D(8, (8,8), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(16, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

- Worse performance

- Try add back the number of convolutions

#### Attempt 4:

14/14 [==============================] - 2s 130ms/step - loss: 70.2126 - mean_absolute_percentage_error: 70.2126 - val_loss: 71.1285 - val_mean_absolute_percentage_error: 71.1285 Epoch 99/100 14/14 [==============================] - 2s 131ms/step - loss: 70.4389 - mean_absolute_percentage_error: 70.4389 - val_loss: 68.8873 - val_mean_absolute_percentage_error: 68.8873 Epoch 100/100 14/14 [==============================] - 2s 130ms/step - loss: 70.2478 - mean_absolute_percentage_error: 70.2478 - val_loss: 69.1277 - val_mean_absolute_percentage_error: 69.1277

```python
model = Sequential()
model.add(Conv2D(16, (8,8), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.add(Activation('softmax'))
model.add(Dense(1))
```

- Extremely good generalisation 

- Not great accuracy

- Try increasing complexity of model

#### Attempt 5:

14/14 [==============================] - 2s 125ms/step - loss: 9.1037 - mean_absolute_percentage_error: 9.1037 - val_loss: 69.9913 - val_mean_absolute_percentage_error: 69.9913 Epoch 99/100 14/14 [==============================] - 2s 126ms/step - loss: 9.9804 - mean_absolute_percentage_error: 9.9804 - val_loss: 79.2479 - val_mean_absolute_percentage_error: 79.2479 Epoch 100/100 14/14 [==============================] - 2s 126ms/step - loss: 9.1648 - mean_absolute_percentage_error: 9.1648 - val_loss: 76.6723 - val_mean_absolute_percentage_error: 76.6723

```python
model = Sequential()
model.add(Conv2D(16, (8,8), padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(32, (6,6), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (6,6), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16))
model.add(Activation('softmax'))
model.add(Dense(1))
```
