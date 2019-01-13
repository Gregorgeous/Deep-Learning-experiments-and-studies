# My first experiment:
---
# Generalisation vs. efficiency dilemma on the example of the binary classification problem. 
---

## Section 1: Introduction:

This document compares two deep learning models that aim to recognise the sentiment (i.e. positivity or negativity) of a movie review. The first one is the model presented on the lectures introducing the Binary classification problem (Jupyter Notebook 3.4:classifying-movie-reviews-a-binary-classification-example) and the second one is an enhanced version using regularisers, the drop-out layers and different hyper-parameters added incrementally after running a series of training sessions and plotting their results.   

---
### _Main objective_: discover if, when and why the second model is better than the first one. 
---

My comparison is conducted on the IMDB dataset described in the "Deep Learning With Python" book (Chollet F., Manning Publications)[1] that is available directly in the Keras API.

### 1.1 Document structure 

The document is structured into 3 sections. Section 1 is an introduction. Section 2 provides the reader with the results of training both the base and the enhanced model found respectively in parts 2.1 and 2.2. Additionally, whenever there is any new concept that was not described in this document, there will be an explanation of it prior to running the adequate code.   
Section 3 gives a conclusion to the experiment along with some extra remarks that didn't fit find space in section 2. 

---

## Section 2: IMDB Dataset deep learning models. Methods and results.

### 2.1: Overview and base implementation 

##### 2.1.1: IMDB Dataset description

IMDB Dataset is a collection of reviews from the Internet Movie Database. It consists of 50,000 movie reviews that are split into 2 sets 25,000 reviews each, one for training and one for testing purposes. Each review has its label (often called a "sentiment"). If the review is positive it has a label "1", otherwise "0". Being given 25,000 reviews, our model will try to learn associations between used words in the reviews and the sentiment this particular review has. In simpler terms, given a large number of reviews with a label of "1" (i.e. positive) such as e.g. "I like this movie", "It is extraordinary" etc. our model will learn that this particular word likely means the review is positive - accordingly it will learn to "feel" what is a negative review. 

##### 2.1.2. Loading the dataset 

IMDB movie reviews dataset is one of the datasets that comes within Keras' API. It has been preprocessed for a data scientist's convenience and therefore saves a number of steps otherwise necessary during the pre-processing stage of the dataset. The list of those is following: [2]

- Instead of just a sequence of strings, each review was encoded into a sequence of word indices. Moreover, they are encoded in such a way the most frequent words are the ones with the lowest index. 
- Upon importing the imdb dataset from keras and calling imdb.load_data in python code, will return to us 2 tuples, first with a pair of the training reviews and test reviews and the second with a pair of associated training labels and test labels (see the cell below).
- 8 optional arguments during the imdb.load_data() call which allow us e.g. to limit our encoded words to X number of the most frequent words ("num_words" argument) so that any less frequent word will be encoded as "unknown word" (by default the index for an unknown word is "2" and it can be changed with another optional argument "oov_char") or ignore a desired number of the most frequent words ("skip_top" argument) - this will also encode those top words with the index of the unknown word.   

Let us load the dataset in code:


```python
from keras.datasets import imdb

((train_data, train_labels), (test_data, test_labels)) = imdb.load_data(num_words = 10000)
```

    Using TensorFlow backend.
    

###### 2.1.3: Pre-processing stage

One of the harder parts of creating a real-life deep learning model is to convert the data into a format the model can properly make calculations on and successively alter the data of. In order for the model to do that the data needs to be fed to the model in the shape of vectorised tensors. The imdb dataset we acquire from keras already comes mostly pre-processed and the only task of ours is to vectorise the given data.    

For that we will use a piece of code provided on the lecture: 


```python
import numpy as np

def vectorise_sequences(sequences, dimension = 10000):
    results = np.zeros( (len(sequences), dimension) )
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

```

Following this "DRY" convention , we might want to modify the lecture code and also create "vectorise_labels" function, we might not save a lot of code lines, but we will at least we increased the code readability by properly naming the function. Because the labels already come as a tuple conformed of 0-1 integers, we just simply need to return them as a float32-encoded array (or "list" in python's terms).   


```python
def vectorise_labels (raw_labels):
    return np.asarray(raw_labels).astype('float32')

y_train = vectorise_labels(train_labels) 
y_test =  vectorise_labels(test_labels)
```

###### 2.1.4: Composing the model and the validation set


```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```

We will use a hold-out validation approach. This approach will be later evaluated on terms of whether it will or will not make  an information leak.  


```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
```

###### 2.1.5: Training the model 


```python
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 6s 401us/step - loss: 0.5055 - acc: 0.7863 - val_loss: 0.3782 - val_acc: 0.8702
    Epoch 2/20
    15000/15000 [==============================] - 4s 262us/step - loss: 0.2994 - acc: 0.9049 - val_loss: 0.3002 - val_acc: 0.8897
    Epoch 3/20
    15000/15000 [==============================] - 3s 228us/step - loss: 0.2174 - acc: 0.9288 - val_loss: 0.3086 - val_acc: 0.8714
    Epoch 4/20
    15000/15000 [==============================] - 3s 176us/step - loss: 0.1748 - acc: 0.9438 - val_loss: 0.2831 - val_acc: 0.8844
    Epoch 5/20
    15000/15000 [==============================] - 3s 183us/step - loss: 0.1422 - acc: 0.9543 - val_loss: 0.2856 - val_acc: 0.8858
    Epoch 6/20
    15000/15000 [==============================] - 3s 215us/step - loss: 0.1149 - acc: 0.9653 - val_loss: 0.3128 - val_acc: 0.8787
    Epoch 7/20
    15000/15000 [==============================] - 3s 182us/step - loss: 0.0977 - acc: 0.9711 - val_loss: 0.3133 - val_acc: 0.8842
    Epoch 8/20
    15000/15000 [==============================] - 3s 168us/step - loss: 0.0806 - acc: 0.9763 - val_loss: 0.3860 - val_acc: 0.8659
    Epoch 9/20
    15000/15000 [==============================] - 3s 181us/step - loss: 0.0661 - acc: 0.9819 - val_loss: 0.3639 - val_acc: 0.8780
    Epoch 10/20
    15000/15000 [==============================] - 3s 183us/step - loss: 0.0565 - acc: 0.9847 - val_loss: 0.3853 - val_acc: 0.8792
    Epoch 11/20
    15000/15000 [==============================] - 3s 187us/step - loss: 0.0426 - acc: 0.9906 - val_loss: 0.4131 - val_acc: 0.8781
    Epoch 12/20
    15000/15000 [==============================] - 4s 235us/step - loss: 0.0375 - acc: 0.9921 - val_loss: 0.4611 - val_acc: 0.8683
    Epoch 13/20
    15000/15000 [==============================] - 3s 176us/step - loss: 0.0299 - acc: 0.9928 - val_loss: 0.4719 - val_acc: 0.8732
    Epoch 14/20
    15000/15000 [==============================] - 3s 184us/step - loss: 0.0249 - acc: 0.9945 - val_loss: 0.5059 - val_acc: 0.8716
    Epoch 15/20
    15000/15000 [==============================] - 3s 176us/step - loss: 0.0197 - acc: 0.9965 - val_loss: 0.5328 - val_acc: 0.8715
    Epoch 16/20
    15000/15000 [==============================] - 3s 182us/step - loss: 0.0149 - acc: 0.9980 - val_loss: 0.5695 - val_acc: 0.8683
    Epoch 17/20
    15000/15000 [==============================] - 3s 184us/step - loss: 0.0141 - acc: 0.9980 - val_loss: 0.6011 - val_acc: 0.8663
    Epoch 18/20
    15000/15000 [==============================] - 3s 202us/step - loss: 0.0095 - acc: 0.9991 - val_loss: 0.6334 - val_acc: 0.8666
    Epoch 19/20
    15000/15000 [==============================] - 3s 204us/step - loss: 0.0077 - acc: 0.9990 - val_loss: 0.7221 - val_acc: 0.8564
    Epoch 20/20
    15000/15000 [==============================] - 2s 163us/step - loss: 0.0047 - acc: 0.9999 - val_loss: 0.6886 - val_acc: 0.8663
    

###### 2.1.6: Plotting and the evaluation of the training results

As we will be plotting our accuracy and loss results many time over this experiment, let us enhance the original code provided on the lectures and create a more "DRY" approach of the plotting procedure by creating a function definition "plotFunc" : 


```python
import matplotlib.pyplot as plt
def plotFunc(isLossPlot, epochsNum, train_LossOrAcc, val_LossOrAcc):
    """Plots either a train & val loss (when isLossPlot is True) or the accuracy"""
    plt.clf() #Clear any previous plot
    if isLossPlot: 
        label1 = 'Training loss'
        label2 = 'Validation loss'
    else: 
        label1 = 'Training accuracy'
        label2 = 'Validation accuracy'
    plt.plot(epochsNum, train_LossOrAcc, 'bo', label =label1)
    plt.plot(epochsNum, val_LossOrAcc, 'b', label = label2)  
    plt.xlabel('Epochs')
    # We repeat the condition to correctly name the ylabel. 
    # NB. We could avoid repeating the conditional by just extracting/slicing the second word from the label 1 or 2 
    # but for an increased readability I left it this way. 
    if isLossPlot:
        plt.title("Training and validation loss")
        plt.ylabel('Loss')
    else: 
        plt.title("Training and validation accuracy")
        plt.ylabel('Accuracy')
    plt.legend()


```

Our next step is outputting the results of the loss and accuracy results of our model on both the training and validation sets as a plot. For that we will firstly obtain necessary data from our model's memory that can by found in a Python dictionary's format by calling: "our_model_variable_name.history". We will use our newly created plotFunc function to handle the plotting action.


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_34_0.png)


Similarly we will now plot the accuracy results:


```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_36_0.png)


### 2.2: Enhancing the model

##### 2.2.1: Adding a regulariser 

Using both L1 and L2 regularisers


```python
from keras import regularizers
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```


```python
ourModel = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 4s 282us/step - loss: 1.0964 - acc: 0.7569 - val_loss: 0.7484 - val_acc: 0.8266
    Epoch 2/20
    15000/15000 [==============================] - 4s 236us/step - loss: 0.6994 - acc: 0.8285 - val_loss: 0.6686 - val_acc: 0.8487
    Epoch 3/20
    15000/15000 [==============================] - 3s 191us/step - loss: 0.6348 - acc: 0.8533 - val_loss: 0.6566 - val_acc: 0.8226
    Epoch 4/20
    15000/15000 [==============================] - 3s 211us/step - loss: 0.6009 - acc: 0.8545 - val_loss: 0.6011 - val_acc: 0.8563
    Epoch 5/20
    15000/15000 [==============================] - 3s 183us/step - loss: 0.5743 - acc: 0.8659 - val_loss: 0.5802 - val_acc: 0.8633
    Epoch 6/20
    15000/15000 [==============================] - 3s 169us/step - loss: 0.5635 - acc: 0.8645 - val_loss: 0.5719 - val_acc: 0.8660
    Epoch 7/20
    15000/15000 [==============================] - 3s 186us/step - loss: 0.5518 - acc: 0.8681 - val_loss: 0.5607 - val_acc: 0.8662
    Epoch 8/20
    15000/15000 [==============================] - 3s 171us/step - loss: 0.5437 - acc: 0.8684 - val_loss: 0.5489 - val_acc: 0.8692
    Epoch 9/20
    15000/15000 [==============================] - 3s 167us/step - loss: 0.5315 - acc: 0.8766 - val_loss: 0.5599 - val_acc: 0.8594
    Epoch 10/20
    15000/15000 [==============================] - 2s 161us/step - loss: 0.5256 - acc: 0.8749 - val_loss: 0.5385 - val_acc: 0.8703
    Epoch 11/20
    15000/15000 [==============================] - 2s 165us/step - loss: 0.5233 - acc: 0.8761 - val_loss: 0.5485 - val_acc: 0.8648
    Epoch 12/20
    15000/15000 [==============================] - 2s 161us/step - loss: 0.5107 - acc: 0.8815 - val_loss: 0.5343 - val_acc: 0.8721
    Epoch 13/20
    15000/15000 [==============================] - 3s 174us/step - loss: 0.5095 - acc: 0.8783 - val_loss: 0.5303 - val_acc: 0.8740
    Epoch 14/20
    15000/15000 [==============================] - 3s 179us/step - loss: 0.5075 - acc: 0.8804 - val_loss: 0.5318 - val_acc: 0.8734
    Epoch 15/20
    15000/15000 [==============================] - 2s 166us/step - loss: 0.4997 - acc: 0.8835 - val_loss: 0.5252 - val_acc: 0.8761
    Epoch 16/20
    15000/15000 [==============================] - 2s 160us/step - loss: 0.4967 - acc: 0.8860 - val_loss: 0.5280 - val_acc: 0.8691
    Epoch 17/20
    15000/15000 [==============================] - 2s 165us/step - loss: 0.4927 - acc: 0.8851 - val_loss: 0.5275 - val_acc: 0.8729
    Epoch 18/20
    15000/15000 [==============================] - 2s 161us/step - loss: 0.4851 - acc: 0.8898 - val_loss: 0.5174 - val_acc: 0.8754
    Epoch 19/20
    15000/15000 [==============================] - 2s 165us/step - loss: 0.4823 - acc: 0.8911 - val_loss: 0.5140 - val_acc: 0.8773
    Epoch 20/20
    15000/15000 [==============================] - 3s 190us/step - loss: 0.4784 - acc: 0.8925 - val_loss: 0.5121 - val_acc: 0.8784
    


```python
history_dict = ourModel.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_43_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_44_0.png)


Using the L2 regulariser only


```python
from keras import regularizers
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu',  kernel_regularizer=regularizers.l2(0.001), input_shape = (10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```


```python
ourModel = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 3s 227us/step - loss: 0.5615 - acc: 0.7843 - val_loss: 0.4348 - val_acc: 0.8676
    Epoch 2/20
    15000/15000 [==============================] - 3s 205us/step - loss: 0.3661 - acc: 0.8933 - val_loss: 0.3663 - val_acc: 0.8792
    Epoch 3/20
    15000/15000 [==============================] - 3s 204us/step - loss: 0.2928 - acc: 0.9191 - val_loss: 0.3323 - val_acc: 0.8909
    Epoch 4/20
    15000/15000 [==============================] - 3s 185us/step - loss: 0.2556 - acc: 0.9309 - val_loss: 0.3425 - val_acc: 0.8802
    Epoch 5/20
    15000/15000 [==============================] - 3s 214us/step - loss: 0.2320 - acc: 0.9398 - val_loss: 0.3314 - val_acc: 0.8874
    Epoch 6/20
    15000/15000 [==============================] - 3s 193us/step - loss: 0.2126 - acc: 0.9485 - val_loss: 0.3380 - val_acc: 0.8852
    Epoch 7/20
    15000/15000 [==============================] - 3s 213us/step - loss: 0.2059 - acc: 0.9482 - val_loss: 0.3419 - val_acc: 0.8851
    Epoch 8/20
    15000/15000 [==============================] - 4s 248us/step - loss: 0.1924 - acc: 0.9566 - val_loss: 0.3534 - val_acc: 0.8828
    Epoch 9/20
    15000/15000 [==============================] - 3s 225us/step - loss: 0.1876 - acc: 0.9577 - val_loss: 0.3816 - val_acc: 0.8725
    Epoch 10/20
    15000/15000 [==============================] - 3s 226us/step - loss: 0.1745 - acc: 0.9630 - val_loss: 0.3676 - val_acc: 0.8815
    Epoch 11/20
    15000/15000 [==============================] - 3s 226us/step - loss: 0.1750 - acc: 0.9604 - val_loss: 0.3970 - val_acc: 0.8746
    Epoch 12/20
    15000/15000 [==============================] - 4s 282us/step - loss: 0.1679 - acc: 0.9658 - val_loss: 0.3844 - val_acc: 0.8800
    Epoch 13/20
    15000/15000 [==============================] - 3s 205us/step - loss: 0.1651 - acc: 0.9665 - val_loss: 0.4187 - val_acc: 0.8725
    Epoch 14/20
    15000/15000 [==============================] - 3s 203us/step - loss: 0.1624 - acc: 0.9667 - val_loss: 0.3978 - val_acc: 0.8790
    Epoch 15/20
    15000/15000 [==============================] - 3s 193us/step - loss: 0.1555 - acc: 0.9708 - val_loss: 0.4855 - val_acc: 0.8586
    Epoch 16/20
    15000/15000 [==============================] - 4s 243us/step - loss: 0.1512 - acc: 0.9721 - val_loss: 0.4207 - val_acc: 0.8694
    Epoch 17/20
    15000/15000 [==============================] - 4s 284us/step - loss: 0.1506 - acc: 0.9719 - val_loss: 0.4483 - val_acc: 0.8628
    Epoch 18/20
    15000/15000 [==============================] - 3s 174us/step - loss: 0.1493 - acc: 0.9722 - val_loss: 0.4483 - val_acc: 0.8649
    Epoch 19/20
    15000/15000 [==============================] - 3s 230us/step - loss: 0.1479 - acc: 0.9710 - val_loss: 0.4339 - val_acc: 0.8724
    Epoch 20/20
    15000/15000 [==============================] - 4s 265us/step - loss: 0.1391 - acc: 0.9769 - val_loss: 0.4408 - val_acc: 0.8717
    


```python
history_dict = ourModel.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_49_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_50_0.png)


Regulariser choice verdict: **L1_L2**. 


Reasoning: as shown on the plots above, using both L1 and L2 regularisers gives a promise of better results and much later occurring over-fitting   

##### 2.2.2. Adding a dropout 


```python
from keras import regularizers
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
ourModel = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 4s 259us/step - loss: 1.1881 - acc: 0.6158 - val_loss: 0.8311 - val_acc: 0.7355
    Epoch 2/20
    15000/15000 [==============================] - 3s 191us/step - loss: 0.7822 - acc: 0.7185 - val_loss: 0.7329 - val_acc: 0.8446
    Epoch 3/20
    15000/15000 [==============================] - 3s 226us/step - loss: 0.7366 - acc: 0.7670 - val_loss: 0.6737 - val_acc: 0.8519
    Epoch 4/20
    15000/15000 [==============================] - 3s 227us/step - loss: 0.7075 - acc: 0.7878 - val_loss: 0.6525 - val_acc: 0.8473
    Epoch 5/20
    15000/15000 [==============================] - 3s 205us/step - loss: 0.6868 - acc: 0.8067 - val_loss: 0.6091 - val_acc: 0.8583
    Epoch 6/20
    15000/15000 [==============================] - 3s 213us/step - loss: 0.6729 - acc: 0.8128 - val_loss: 0.5894 - val_acc: 0.8606
    Epoch 7/20
    15000/15000 [==============================] - 3s 213us/step - loss: 0.6523 - acc: 0.8287 - val_loss: 0.5861 - val_acc: 0.8629
    Epoch 8/20
    15000/15000 [==============================] - 4s 238us/step - loss: 0.6446 - acc: 0.8302 - val_loss: 0.5740 - val_acc: 0.8639
    Epoch 9/20
    15000/15000 [==============================] - 3s 193us/step - loss: 0.6403 - acc: 0.8316 - val_loss: 0.5782 - val_acc: 0.8660
    Epoch 10/20
    15000/15000 [==============================] - 3s 186us/step - loss: 0.6342 - acc: 0.8373 - val_loss: 0.5796 - val_acc: 0.8607
    Epoch 11/20
    15000/15000 [==============================] - 3s 183us/step - loss: 0.6246 - acc: 0.8400 - val_loss: 0.5742 - val_acc: 0.8578
    Epoch 12/20
    15000/15000 [==============================] - 3s 192us/step - loss: 0.6246 - acc: 0.8415 - val_loss: 0.5581 - val_acc: 0.8668
    Epoch 13/20
    15000/15000 [==============================] - 3s 204us/step - loss: 0.6169 - acc: 0.8463 - val_loss: 0.5769 - val_acc: 0.8529
    Epoch 14/20
    15000/15000 [==============================] - 3s 220us/step - loss: 0.6201 - acc: 0.8421 - val_loss: 0.5660 - val_acc: 0.8670
    Epoch 15/20
    15000/15000 [==============================] - 4s 247us/step - loss: 0.6153 - acc: 0.8450 - val_loss: 0.5670 - val_acc: 0.8660
    Epoch 16/20
    15000/15000 [==============================] - 4s 244us/step - loss: 0.6108 - acc: 0.8498 - val_loss: 0.5718 - val_acc: 0.8681
    Epoch 17/20
    15000/15000 [==============================] - 3s 208us/step - loss: 0.6118 - acc: 0.8491 - val_loss: 0.5522 - val_acc: 0.8703
    Epoch 18/20
    15000/15000 [==============================] - 3s 228us/step - loss: 0.6134 - acc: 0.8495 - val_loss: 0.5543 - val_acc: 0.8712
    Epoch 19/20
    15000/15000 [==============================] - 3s 196us/step - loss: 0.6146 - acc: 0.8455 - val_loss: 0.5598 - val_acc: 0.8671
    Epoch 20/20
    15000/15000 [==============================] - 3s 187us/step - loss: 0.6053 - acc: 0.8523 - val_loss: 0.6104 - val_acc: 0.8308
    


```python
history_dict = ourModel.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_56_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_57_0.png)


##### 2.2.3: Adjusting the hyper-parameters 

##### Smaller network's size


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
ourModel = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 4s 253us/step - loss: 0.8088 - acc: 0.5473 - val_loss: 0.7136 - val_acc: 0.6370
    Epoch 2/20
    15000/15000 [==============================] - 3s 209us/step - loss: 0.7093 - acc: 0.5813 - val_loss: 0.6929 - val_acc: 0.7891
    Epoch 3/20
    15000/15000 [==============================] - 3s 179us/step - loss: 0.7006 - acc: 0.5986 - val_loss: 0.6784 - val_acc: 0.7142
    Epoch 4/20
    15000/15000 [==============================] - 3s 168us/step - loss: 0.6966 - acc: 0.6003 - val_loss: 0.6695 - val_acc: 0.7718
    Epoch 5/20
    15000/15000 [==============================] - 3s 182us/step - loss: 0.6916 - acc: 0.6103 - val_loss: 0.6739 - val_acc: 0.8420
    Epoch 6/20
    15000/15000 [==============================] - 3s 197us/step - loss: 0.6888 - acc: 0.6163 - val_loss: 0.6682 - val_acc: 0.8460
    Epoch 7/20
    15000/15000 [==============================] - 3s 170us/step - loss: 0.6875 - acc: 0.6177 - val_loss: 0.6545 - val_acc: 0.8418
    Epoch 8/20
    15000/15000 [==============================] - 3s 187us/step - loss: 0.6825 - acc: 0.6256 - val_loss: 0.6550 - val_acc: 0.8426
    Epoch 9/20
    15000/15000 [==============================] - 3s 173us/step - loss: 0.6790 - acc: 0.6317 - val_loss: 0.6513 - val_acc: 0.8382
    Epoch 10/20
    15000/15000 [==============================] - 3s 167us/step - loss: 0.6765 - acc: 0.6387 - val_loss: 0.6217 - val_acc: 0.8291
    Epoch 11/20
    15000/15000 [==============================] - 3s 186us/step - loss: 0.6784 - acc: 0.6395 - val_loss: 0.6238 - val_acc: 0.8483
    Epoch 12/20
    15000/15000 [==============================] - 3s 189us/step - loss: 0.6757 - acc: 0.6486 - val_loss: 0.6473 - val_acc: 0.8218
    Epoch 13/20
    15000/15000 [==============================] - 3s 167us/step - loss: 0.6707 - acc: 0.6595 - val_loss: 0.6272 - val_acc: 0.8439
    Epoch 14/20
    15000/15000 [==============================] - 3s 198us/step - loss: 0.6646 - acc: 0.6681 - val_loss: 0.6071 - val_acc: 0.8528
    Epoch 15/20
    15000/15000 [==============================] - 3s 176us/step - loss: 0.6670 - acc: 0.6675 - val_loss: 0.6178 - val_acc: 0.8407
    Epoch 16/20
    15000/15000 [==============================] - 3s 167us/step - loss: 0.6639 - acc: 0.6711 - val_loss: 0.5933 - val_acc: 0.8553
    Epoch 17/20
    15000/15000 [==============================] - 3s 180us/step - loss: 0.6614 - acc: 0.6746 - val_loss: 0.6097 - val_acc: 0.8411
    Epoch 18/20
    15000/15000 [==============================] - 3s 200us/step - loss: 0.6595 - acc: 0.6764 - val_loss: 0.5901 - val_acc: 0.8542
    Epoch 19/20
    15000/15000 [==============================] - 3s 181us/step - loss: 0.6594 - acc: 0.6773 - val_loss: 0.5909 - val_acc: 0.8540
    Epoch 20/20
    15000/15000 [==============================] - 3s 176us/step - loss: 0.6553 - acc: 0.6830 - val_loss: 0.5710 - val_acc: 0.8518
    


```python
history_dict = ourModel.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_62_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_63_0.png)


##### Increased Learning Rate


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.002),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 3s 231us/step - loss: 0.7926 - acc: 0.5049 - val_loss: 0.7640 - val_acc: 0.4947
    Epoch 2/20
    15000/15000 [==============================] - 3s 198us/step - loss: 0.7434 - acc: 0.5451 - val_loss: 0.7163 - val_acc: 0.8119
    Epoch 3/20
    15000/15000 [==============================] - 3s 181us/step - loss: 0.7208 - acc: 0.5931 - val_loss: 0.6922 - val_acc: 0.7248
    Epoch 4/20
    15000/15000 [==============================] - 3s 185us/step - loss: 0.7055 - acc: 0.6060 - val_loss: 0.6548 - val_acc: 0.8386
    Epoch 5/20
    15000/15000 [==============================] - 3s 196us/step - loss: 0.6975 - acc: 0.6513 - val_loss: 0.6482 - val_acc: 0.8285
    Epoch 6/20
    15000/15000 [==============================] - 3s 169us/step - loss: 0.6892 - acc: 0.6715 - val_loss: 0.6801 - val_acc: 0.7351
    Epoch 7/20
    15000/15000 [==============================] - 3s 168us/step - loss: 0.6856 - acc: 0.6751 - val_loss: 0.6098 - val_acc: 0.8501
    Epoch 8/20
    15000/15000 [==============================] - 3s 212us/step - loss: 0.6807 - acc: 0.6843 - val_loss: 0.6090 - val_acc: 0.8348
    Epoch 9/20
    15000/15000 [==============================] - 4s 239us/step - loss: 0.6808 - acc: 0.6882 - val_loss: 0.5835 - val_acc: 0.8625
    Epoch 10/20
    15000/15000 [==============================] - 4s 254us/step - loss: 0.6725 - acc: 0.6947 - val_loss: 0.5986 - val_acc: 0.8376
    Epoch 11/20
    15000/15000 [==============================] - 3s 211us/step - loss: 0.6786 - acc: 0.6928 - val_loss: 0.5740 - val_acc: 0.8632
    Epoch 12/20
    15000/15000 [==============================] - 3s 194us/step - loss: 0.6799 - acc: 0.6938 - val_loss: 0.5682 - val_acc: 0.8571
    Epoch 13/20
    15000/15000 [==============================] - 3s 184us/step - loss: 0.6726 - acc: 0.6973 - val_loss: 0.5673 - val_acc: 0.8641
    Epoch 14/20
    15000/15000 [==============================] - 4s 261us/step - loss: 0.6740 - acc: 0.7025 - val_loss: 0.5665 - val_acc: 0.8649
    Epoch 15/20
    15000/15000 [==============================] - 3s 222us/step - loss: 0.6759 - acc: 0.6989 - val_loss: 0.5726 - val_acc: 0.8571
    Epoch 16/20
    15000/15000 [==============================] - 3s 226us/step - loss: 0.6723 - acc: 0.7013 - val_loss: 0.5661 - val_acc: 0.8393
    Epoch 17/20
    15000/15000 [==============================] - 3s 195us/step - loss: 0.6783 - acc: 0.6968 - val_loss: 0.5711 - val_acc: 0.8631
    Epoch 18/20
    15000/15000 [==============================] - 3s 184us/step - loss: 0.6769 - acc: 0.6959 - val_loss: 0.5607 - val_acc: 0.8662
    Epoch 19/20
    15000/15000 [==============================] - 3s 213us/step - loss: 0.6775 - acc: 0.6967 - val_loss: 0.5670 - val_acc: 0.8652
    Epoch 20/20
    15000/15000 [==============================] - 3s 203us/step - loss: 0.6733 - acc: 0.7027 - val_loss: 0.5717 - val_acc: 0.8610
    


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_67_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_68_0.png)


Verdict: an increased learning rate causes a bigger spread of the results. As our model seemed not to over-fit over the epochs when the learning rate was smaller I believe there is no need to increase the learning rate.

##### Various Batch Sizes 

Batch size 256


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 256,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 5s 333us/step - loss: 0.7734 - acc: 0.5837 - val_loss: 0.7125 - val_acc: 0.7767
    Epoch 2/20
    15000/15000 [==============================] - 3s 195us/step - loss: 0.6942 - acc: 0.6655 - val_loss: 0.6512 - val_acc: 0.8369
    Epoch 3/20
    15000/15000 [==============================] - 3s 196us/step - loss: 0.6707 - acc: 0.7002 - val_loss: 0.6152 - val_acc: 0.8500
    Epoch 4/20
    15000/15000 [==============================] - 3s 186us/step - loss: 0.6505 - acc: 0.7168 - val_loss: 0.5846 - val_acc: 0.8311
    Epoch 5/20
    15000/15000 [==============================] - 3s 198us/step - loss: 0.6390 - acc: 0.7199 - val_loss: 0.5683 - val_acc: 0.8594
    Epoch 6/20
    15000/15000 [==============================] - 4s 285us/step - loss: 0.6353 - acc: 0.7266 - val_loss: 0.5524 - val_acc: 0.8564
    Epoch 7/20
    15000/15000 [==============================] - 5s 303us/step - loss: 0.6293 - acc: 0.7344 - val_loss: 0.5361 - val_acc: 0.8566
    Epoch 8/20
    15000/15000 [==============================] - 3s 196us/step - loss: 0.6239 - acc: 0.7345 - val_loss: 0.5343 - val_acc: 0.8609
    Epoch 9/20
    15000/15000 [==============================] - 3s 194us/step - loss: 0.6183 - acc: 0.7359 - val_loss: 0.5362 - val_acc: 0.8599
    Epoch 10/20
    15000/15000 [==============================] - 3s 223us/step - loss: 0.6181 - acc: 0.7397 - val_loss: 0.5171 - val_acc: 0.8648
    Epoch 11/20
    15000/15000 [==============================] - 3s 222us/step - loss: 0.6162 - acc: 0.7453 - val_loss: 0.5195 - val_acc: 0.8624
    Epoch 12/20
    15000/15000 [==============================] - 3s 193us/step - loss: 0.6162 - acc: 0.7473 - val_loss: 0.5127 - val_acc: 0.8637
    Epoch 13/20
    15000/15000 [==============================] - 3s 193us/step - loss: 0.6131 - acc: 0.7479 - val_loss: 0.5171 - val_acc: 0.8606
    Epoch 14/20
    15000/15000 [==============================] - 3s 195us/step - loss: 0.6097 - acc: 0.7517 - val_loss: 0.5082 - val_acc: 0.8633
    Epoch 15/20
    15000/15000 [==============================] - 3s 197us/step - loss: 0.6105 - acc: 0.7487 - val_loss: 0.5061 - val_acc: 0.8606
    Epoch 16/20
    15000/15000 [==============================] - 3s 227us/step - loss: 0.6148 - acc: 0.7472 - val_loss: 0.5210 - val_acc: 0.8569
    Epoch 17/20
    15000/15000 [==============================] - 3s 195us/step - loss: 0.6023 - acc: 0.7582 - val_loss: 0.5049 - val_acc: 0.8662
    Epoch 18/20
    15000/15000 [==============================] - 3s 209us/step - loss: 0.6093 - acc: 0.7533 - val_loss: 0.5058 - val_acc: 0.8648
    Epoch 19/20
    15000/15000 [==============================] - 3s 220us/step - loss: 0.6055 - acc: 0.7507 - val_loss: 0.5065 - val_acc: 0.8660
    Epoch 20/20
    15000/15000 [==============================] - 3s 217us/step - loss: 0.6043 - acc: 0.7572 - val_loss: 0.5035 - val_acc: 0.8659
    


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_74_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_75_0.png)


Batch size 1024


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))
```


```python
from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 20,
                    batch_size = 1024,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/20
    15000/15000 [==============================] - 4s 273us/step - loss: 0.9180 - acc: 0.5343 - val_loss: 0.7503 - val_acc: 0.7709
    Epoch 2/20
    15000/15000 [==============================] - 3s 173us/step - loss: 0.7247 - acc: 0.5912 - val_loss: 0.6780 - val_acc: 0.8249
    Epoch 3/20
    15000/15000 [==============================] - 2s 157us/step - loss: 0.6907 - acc: 0.6039 - val_loss: 0.6529 - val_acc: 0.8455
    Epoch 4/20
    15000/15000 [==============================] - 2s 155us/step - loss: 0.6785 - acc: 0.6138 - val_loss: 0.6369 - val_acc: 0.8321
    Epoch 5/20
    15000/15000 [==============================] - 3s 172us/step - loss: 0.6697 - acc: 0.6169 - val_loss: 0.6157 - val_acc: 0.8543
    Epoch 6/20
    15000/15000 [==============================] - 3s 171us/step - loss: 0.6643 - acc: 0.6237 - val_loss: 0.6006 - val_acc: 0.8563
    Epoch 7/20
    15000/15000 [==============================] - 3s 189us/step - loss: 0.6528 - acc: 0.6362 - val_loss: 0.5887 - val_acc: 0.8604
    Epoch 8/20
    15000/15000 [==============================] - 2s 164us/step - loss: 0.6485 - acc: 0.6481 - val_loss: 0.6024 - val_acc: 0.8381
    Epoch 9/20
    15000/15000 [==============================] - 3s 171us/step - loss: 0.6433 - acc: 0.6511 - val_loss: 0.5623 - val_acc: 0.8641
    Epoch 10/20
    15000/15000 [==============================] - 3s 169us/step - loss: 0.6452 - acc: 0.6572 - val_loss: 0.5635 - val_acc: 0.8642
    Epoch 11/20
    15000/15000 [==============================] - 3s 175us/step - loss: 0.6373 - acc: 0.6651 - val_loss: 0.5564 - val_acc: 0.8634
    Epoch 12/20
    15000/15000 [==============================] - 3s 194us/step - loss: 0.6375 - acc: 0.6765 - val_loss: 0.5549 - val_acc: 0.8647
    Epoch 13/20
    15000/15000 [==============================] - 3s 167us/step - loss: 0.6367 - acc: 0.6759 - val_loss: 0.5603 - val_acc: 0.8643
    Epoch 14/20
    15000/15000 [==============================] - 3s 182us/step - loss: 0.6334 - acc: 0.6887 - val_loss: 0.5458 - val_acc: 0.8598
    Epoch 15/20
    15000/15000 [==============================] - 3s 175us/step - loss: 0.6310 - acc: 0.6897 - val_loss: 0.5441 - val_acc: 0.8571
    Epoch 16/20
    15000/15000 [==============================] - 3s 172us/step - loss: 0.6263 - acc: 0.7005 - val_loss: 0.5399 - val_acc: 0.8560
    Epoch 17/20
    15000/15000 [==============================] - 3s 178us/step - loss: 0.6284 - acc: 0.6997 - val_loss: 0.5373 - val_acc: 0.8670
    Epoch 18/20
    15000/15000 [==============================] - 3s 215us/step - loss: 0.6252 - acc: 0.7057 - val_loss: 0.5444 - val_acc: 0.8663
    Epoch 19/20
    15000/15000 [==============================] - 3s 169us/step - loss: 0.6238 - acc: 0.7120 - val_loss: 0.5228 - val_acc: 0.8666
    Epoch 20/20
    15000/15000 [==============================] - 3s 173us/step - loss: 0.6201 - acc: 0.7080 - val_loss: 0.5162 - val_acc: 0.8689
    


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_79_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_80_0.png)


Verdict: changing the batch sizes seems to make a marginal difference, therefore I will arbitrarily choose batch size of 256 as it is potentially more data to process (once more, the model seems not to over-fit thus more data to process might give us better generalisation, instead of the contrary if the over-fitting was occurring).

##### Different amount of epochs

Due to the very good results in tackling the overfitting by using the regularisers and a dropout, we might want to increase the number of epochs on our models. Let us see the results of that.

More epochs:


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

```


```python
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 35,
                    batch_size = 256,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/35
    15000/15000 [==============================] - 4s 284us/step - loss: 0.7791 - acc: 0.5209 - val_loss: 0.7004 - val_acc: 0.8050
    Epoch 2/35
    15000/15000 [==============================] - 3s 232us/step - loss: 0.7088 - acc: 0.5487 - val_loss: 0.6781 - val_acc: 0.8318
    Epoch 3/35
    15000/15000 [==============================] - 3s 207us/step - loss: 0.6896 - acc: 0.5803 - val_loss: 0.6649 - val_acc: 0.8451
    Epoch 4/35
    15000/15000 [==============================] - 4s 272us/step - loss: 0.6810 - acc: 0.6073 - val_loss: 0.6456 - val_acc: 0.8331
    Epoch 5/35
    15000/15000 [==============================] - 3s 223us/step - loss: 0.6641 - acc: 0.6213 - val_loss: 0.6194 - val_acc: 0.8388
    Epoch 6/35
    15000/15000 [==============================] - 3s 228us/step - loss: 0.6576 - acc: 0.6213 - val_loss: 0.5822 - val_acc: 0.8584
    Epoch 7/35
    15000/15000 [==============================] - 3s 189us/step - loss: 0.6468 - acc: 0.6434 - val_loss: 0.5733 - val_acc: 0.8583
    Epoch 8/35
    15000/15000 [==============================] - 3s 196us/step - loss: 0.6466 - acc: 0.6409 - val_loss: 0.5641 - val_acc: 0.8597
    Epoch 9/35
    15000/15000 [==============================] - 3s 204us/step - loss: 0.6482 - acc: 0.6480 - val_loss: 0.5651 - val_acc: 0.8575
    Epoch 10/35
    15000/15000 [==============================] - 3s 201us/step - loss: 0.6424 - acc: 0.6475 - val_loss: 0.5382 - val_acc: 0.8637
    Epoch 11/35
    15000/15000 [==============================] - 4s 295us/step - loss: 0.6419 - acc: 0.6481 - val_loss: 0.5384 - val_acc: 0.8640
    Epoch 12/35
    15000/15000 [==============================] - 3s 231us/step - loss: 0.6434 - acc: 0.6477 - val_loss: 0.5474 - val_acc: 0.8627
    Epoch 13/35
    15000/15000 [==============================] - 3s 216us/step - loss: 0.6374 - acc: 0.6553 - val_loss: 0.5682 - val_acc: 0.8546
    Epoch 14/35
    15000/15000 [==============================] - 4s 288us/step - loss: 0.6362 - acc: 0.6711 - val_loss: 0.5065 - val_acc: 0.8638
    Epoch 15/35
    15000/15000 [==============================] - 3s 229us/step - loss: 0.6341 - acc: 0.6880 - val_loss: 0.5063 - val_acc: 0.8655
    Epoch 16/35
    15000/15000 [==============================] - 3s 232us/step - loss: 0.6338 - acc: 0.6955 - val_loss: 0.5015 - val_acc: 0.8638
    Epoch 17/35
    15000/15000 [==============================] - 3s 217us/step - loss: 0.6307 - acc: 0.6952 - val_loss: 0.4930 - val_acc: 0.8660
    Epoch 18/35
    15000/15000 [==============================] - 3s 206us/step - loss: 0.6290 - acc: 0.6996 - val_loss: 0.4981 - val_acc: 0.8687
    Epoch 19/35
    15000/15000 [==============================] - 3s 204us/step - loss: 0.6208 - acc: 0.7028 - val_loss: 0.5016 - val_acc: 0.8645
    Epoch 20/35
    15000/15000 [==============================] - 3s 224us/step - loss: 0.6260 - acc: 0.7046 - val_loss: 0.5181 - val_acc: 0.8486
    Epoch 21/35
    15000/15000 [==============================] - 4s 264us/step - loss: 0.6253 - acc: 0.7039 - val_loss: 0.4953 - val_acc: 0.8643
    Epoch 22/35
    15000/15000 [==============================] - 3s 218us/step - loss: 0.6197 - acc: 0.7078 - val_loss: 0.4837 - val_acc: 0.8692
    Epoch 23/35
    15000/15000 [==============================] - 3s 202us/step - loss: 0.6179 - acc: 0.7037 - val_loss: 0.4907 - val_acc: 0.8603
    Epoch 24/35
    15000/15000 [==============================] - 4s 253us/step - loss: 0.6175 - acc: 0.7026 - val_loss: 0.4743 - val_acc: 0.8669
    Epoch 25/35
    15000/15000 [==============================] - 5s 314us/step - loss: 0.6195 - acc: 0.7081 - val_loss: 0.4921 - val_acc: 0.8652
    Epoch 26/35
    15000/15000 [==============================] - 4s 243us/step - loss: 0.6205 - acc: 0.7073 - val_loss: 0.4817 - val_acc: 0.8663
    Epoch 27/35
    15000/15000 [==============================] - 4s 267us/step - loss: 0.6176 - acc: 0.7014 - val_loss: 0.4757 - val_acc: 0.8678
    Epoch 28/35
    15000/15000 [==============================] - 4s 249us/step - loss: 0.6163 - acc: 0.7107 - val_loss: 0.4905 - val_acc: 0.8665
    Epoch 29/35
    15000/15000 [==============================] - 5s 319us/step - loss: 0.6180 - acc: 0.7049 - val_loss: 0.4984 - val_acc: 0.8642
    Epoch 30/35
    15000/15000 [==============================] - 3s 214us/step - loss: 0.6179 - acc: 0.7052 - val_loss: 0.4816 - val_acc: 0.8692
    Epoch 31/35
    15000/15000 [==============================] - 4s 251us/step - loss: 0.6174 - acc: 0.7121 - val_loss: 0.4836 - val_acc: 0.8679
    Epoch 32/35
    15000/15000 [==============================] - 3s 215us/step - loss: 0.6265 - acc: 0.7003 - val_loss: 0.4764 - val_acc: 0.8647
    Epoch 33/35
    15000/15000 [==============================] - 3s 228us/step - loss: 0.6179 - acc: 0.7073 - val_loss: 0.5156 - val_acc: 0.8595
    Epoch 34/35
    15000/15000 [==============================] - 3s 194us/step - loss: 0.6205 - acc: 0.7036 - val_loss: 0.4908 - val_acc: 0.8550
    Epoch 35/35
    15000/15000 [==============================] - 3s 187us/step - loss: 0.6166 - acc: 0.7030 - val_loss: 0.5102 - val_acc: 0.8556
    


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_87_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_88_0.png)


For the sake of a thorough experiment, we can also check the results when there is a smaller number of epochs.

Less epochs: 


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu',  kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

```


```python
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 15,
                    batch_size = 256,
                    validation_data = (x_val, y_val))
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/15
    15000/15000 [==============================] - 4s 287us/step - loss: 0.7659 - acc: 0.5847 - val_loss: 0.6998 - val_acc: 0.7412
    Epoch 2/15
    15000/15000 [==============================] - 4s 268us/step - loss: 0.6994 - acc: 0.6413 - val_loss: 0.6747 - val_acc: 0.7821
    Epoch 3/15
    15000/15000 [==============================] - 3s 210us/step - loss: 0.6835 - acc: 0.6717 - val_loss: 0.6656 - val_acc: 0.8467
    Epoch 4/15
    15000/15000 [==============================] - 3s 209us/step - loss: 0.6729 - acc: 0.6864 - val_loss: 0.6556 - val_acc: 0.8465
    Epoch 5/15
    15000/15000 [==============================] - 3s 225us/step - loss: 0.6645 - acc: 0.6958 - val_loss: 0.6183 - val_acc: 0.8497
    Epoch 6/15
    15000/15000 [==============================] - 3s 212us/step - loss: 0.6530 - acc: 0.7093 - val_loss: 0.6095 - val_acc: 0.8562
    Epoch 7/15
    15000/15000 [==============================] - 3s 214us/step - loss: 0.6509 - acc: 0.7122 - val_loss: 0.6032 - val_acc: 0.8509
    Epoch 8/15
    15000/15000 [==============================] - 3s 214us/step - loss: 0.6484 - acc: 0.7144 - val_loss: 0.5688 - val_acc: 0.8345
    Epoch 9/15
    15000/15000 [==============================] - 4s 249us/step - loss: 0.6431 - acc: 0.7127 - val_loss: 0.5810 - val_acc: 0.8526
    Epoch 10/15
    15000/15000 [==============================] - 3s 214us/step - loss: 0.6369 - acc: 0.7206 - val_loss: 0.5631 - val_acc: 0.8610
    Epoch 11/15
    15000/15000 [==============================] - 4s 247us/step - loss: 0.6369 - acc: 0.7159 - val_loss: 0.5768 - val_acc: 0.8391
    Epoch 12/15
    15000/15000 [==============================] - 3s 180us/step - loss: 0.6342 - acc: 0.7187 - val_loss: 0.5449 - val_acc: 0.8622
    Epoch 13/15
    15000/15000 [==============================] - 3s 202us/step - loss: 0.6368 - acc: 0.7228 - val_loss: 0.5383 - val_acc: 0.8593
    Epoch 14/15
    15000/15000 [==============================] - 3s 199us/step - loss: 0.6303 - acc: 0.7289 - val_loss: 0.5423 - val_acc: 0.8596
    Epoch 15/15
    15000/15000 [==============================] - 4s 236us/step - loss: 0.6255 - acc: 0.7308 - val_loss: 0.5251 - val_acc: 0.8625
    


```python
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```


![png](output_93_0.png)



```python
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_94_0.png)


Verdict: as the results seem a little too optimistic with the over-fitting appearing only slightly in the loss results after almost 30 epochs I begin to worry our model might suffer from an information leak. Due to that, I will, precautiously, choose to stick with the smaller number of 15 epochs. 

##### 2.2.4 : Information leaks   

So far we have seen a rather spectacular effect of applying the regularisers and a dropout; the accuracy and the loss on the plots seem to be greater on the validation data instead of the training data. 
That would mean our model  predicts the reviews' sentiments even better on a never-seen data than on the reviews it has been actually trained. Our model has achieved a tremendous generalisation! But are we sure of it? Not necessarily. One must notice thus far our model has been trained and validated a multiple amount of times already. Although we never trained our model on the validation data (i.e. we never showed the model labels of the validation reviews, only checked if the model's prediction sentiment is correct), the model has been indirectly developing a sense of what answer to provide for a given validation review instead of actually predicting basing on the knowledge gained from the training set. This could be compared to a student who instead of the actual studying does enough of mock exams to develop a sense of examiner's way of creating the exams and, therefore, learning only those questions they know will appear on the exam.                 

One of the natural ways to discover when we happen to have the information leakage is when we seem to have overly optimistic results.

*“too good to be true” performance is “a dead giveaway” of its existence* (“Doing Data Science: Straight Talk from the Frontline” , chapter 13)

With that being said, let us run a test set and see the real-life results 


```python
results = model.evaluate(x_test, y_test)
```

    25000/25000 [==============================] - 11s 457us/step
    


```python
results
```




    [0.5304695237636566, 0.86032]



For a comparison, here is an evaluation of the base model found in part 2.1. This time it has a smaller number of epochs to account for the over-fitting happening after running more than 4 epochs:  


```python
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16 ,activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 4, batch_size = 512)

results = model.evaluate(x_test, y_test)

results
```

    Epoch 1/4
    25000/25000 [==============================] - 5s 192us/step - loss: 0.4901 - acc: 0.8240
    Epoch 2/4
    25000/25000 [==============================] - 4s 146us/step - loss: 0.2801 - acc: 0.9064
    Epoch 3/4
    25000/25000 [==============================] - 3s 137us/step - loss: 0.2070 - acc: 0.9270
    Epoch 4/4
    25000/25000 [==============================] - 3s 126us/step - loss: 0.1710 - acc: 0.9397
    25000/25000 [==============================] - 4s 166us/step
    




    [0.2908405643081665, 0.88516]



## Section 3: Evaluation, conclusion & extra remarks

### 3.1: Enhanced model vs. the original

Our enhanced model, contrary to the predictions, has not suffered from the information leak: the range between 80-90% of the accuracy and 45-55% of the loss on the validation set has proved to be found also on the test set. Should the results be different, we would need to use a k-fold validation approach and might be required to reduce the number of trained models.

However, despite adding regularisers and a dropout as well as changing a number of hyper-parameters, our new model behaves worse than the simpler, original one. A possible explanation of this situation is that we prematurely added regularisers instead of firstly reducing the network's size (as suggested in the DLWP book). This led us to choose a combination of L1 & L2 regularisers instead of just L2 for a promise of more stable, not over-fitting (but ultimately underperforming) results. 

### 3.2: Conclusion: eventuality of a better scalability with the enhanced model  

One important distinction between the models is that the original model achieves its results by finding a "sweet spot" between simply memorizing the training labels and be generalised enough. On the other hand, the enhanced model, due to restricting network's size,  using the regularisers and adding a dropout was forced to learn only the most important patterns and eventually generalise very well: the validation results are mostly better then the ones on the training data. 

The observation above raises a highly interesting possibility: **_given a much bigger and/or a completely different test set it is imaginable that the enhanced model would produce a better result than the original one, because the latter likely was just "lucky enough" to be given a set of data it could output the right result by memorising some information from the training data._**     

This situation, once again, can be compared to a school environment: imagine there are 2 students studying towards a math exam (our test set), the first one tries a mixture of partly understanding the mathematical principles *and* simply memorizing the solution to maths problems that are **likely** to appear on the exam (base model). The second one solely focuses on learning all the the mathematical principles (enhanced model). The first one is fortunate enough to achieve a better score than the latter, hovewer, it is the second one that has a more **general** knowledge and who will be able to achieve more outside the school environment - in the AI scenario: where the test set is so big or different to training data that partial memorizing would perform poorly.             

With the above stated, I believe the question whether to use the basic or the enhanced model depends on the use case. When it is likely the real-world samples (i.e. what our test set mimics) will have highly similar nature or scale, using the basic model would be more efficient. Otherwise, using the enhanced model promises a more **stable** accuracy of 80-90% over the time and scale, where the former is plausibly to fall short on the results.     

###  3.3: Extra remarks: the best configuration

A series of trial and error experiments I have run outside of this notebook made me find a more efficient model using the L2 regulariser that gives a slightly more similar efficiency to the barebones model while also having the generalisation characteritics found in the enhanced model:  


```python
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(8, activation = 'relu',  kernel_regularizer=regularizers.l2(0.0005), input_shape = (10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, kernel_regularizer=regularizers.l2(0.0005), activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(partial_x_train, 
                    partial_y_train,
                    epochs = 6,
                    batch_size = 128,
                    validation_data = (x_val, y_val))
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

plotFunc(True, epochs, loss, val_loss)
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/6
    15000/15000 [==============================] - 4s 285us/step - loss: 0.6417 - acc: 0.6779 - val_loss: 0.5617 - val_acc: 0.7606
    Epoch 2/6
    15000/15000 [==============================] - 3s 217us/step - loss: 0.5490 - acc: 0.7965 - val_loss: 0.4925 - val_acc: 0.8662
    Epoch 3/6
    15000/15000 [==============================] - 3s 216us/step - loss: 0.4907 - acc: 0.8321 - val_loss: 0.4395 - val_acc: 0.8688
    Epoch 4/6
    15000/15000 [==============================] - 4s 246us/step - loss: 0.4531 - acc: 0.8406 - val_loss: 0.4084 - val_acc: 0.8665
    Epoch 5/6
    15000/15000 [==============================] - 3s 210us/step - loss: 0.4168 - acc: 0.8526 - val_loss: 0.3959 - val_acc: 0.8631
    Epoch 6/6
    15000/15000 [==============================] - 3s 209us/step - loss: 0.3814 - acc: 0.8662 - val_loss: 0.3915 - val_acc: 0.8654
    


![png](output_116_1.png)



```python
# Written in a separate cell as the plt.clf() would clear the loss plot if run in the same cell.  
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plotFunc(False, epochs, acc, val_acc)
```


![png](output_117_0.png)



```python
results = model.evaluate(x_test, y_test)
results
```

    25000/25000 [==============================] - 4s 159us/step
    




    [0.39525060079574587, 0.86452]



# Bibliography

[1] Manning Publications, 2017, Chollet F., "Deep Learning with Python"


[2] https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

[3] https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
