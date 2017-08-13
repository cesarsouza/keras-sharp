using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Tests
{
    /// <summary>
    ///   Test for examples in https://keras.io/models/sequential/
    /// </summary>
    /// 
    [TestFixture]
    public class SequentialTest
    {
        [Test]
        public void sequential_example_1()
        {
            #region doc_sequential_example_1
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 500 }));
            model.Add(new Dense(10, activation: new Softmax()));
            model.Compile(optimizer: new RootMeanSquareProp(),
                  loss: new CategoricalCrossEntropy(),
                  metrics: new Accuracy());

            #endregion

            Assert.AreEqual(0, model.trainable_weights);
        }

        [Test]
        public void sequential_guide_1()
        {
            //from keras.models import Sequential
            //from keras.layers import Dense, Activation

            //model = Sequential([
            //    Dense(32, input_shape = (784,)),
            //    Activation('relu'),
            //    Dense(10),
            //    Activation('softmax'),
            //])

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_2()
        {
            //model = Sequential()
            //model.add(Dense(32, input_dim = 784))
            //model.add(Activation('relu'))

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_3()
        {
            //model = Sequential()
            //model.add(Dense(32, input_dim = 784))
            //model.add(Activation('relu'))

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_4()
        {
            //model = Sequential()
            //model.add(Dense(32, input_shape = (784,)))
            //model = Sequential()
            //model.add(Dense(32, input_dim = 784))

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_compilation_1()
        {
            //# For a multi-class classification problem
            //model.compile(optimizer='rmsprop',
            //              loss='categorical_crossentropy',
            //              metrics=['accuracy'])

            //# For a binary classification problem
            //model.compile(optimizer='rmsprop',
            //              loss='binary_crossentropy',
            //              metrics=['accuracy'])

            //# For a mean squared error regression problem
            //model.compile(optimizer='rmsprop',
            //              loss='mse')

            //# For custom metrics
            //import keras.backend as K

            //def mean_pred(y_true, y_pred):
            //    return K.mean(y_pred)

            //model.compile(optimizer='rmsprop',
            //              loss='binary_crossentropy',
            //              metrics=['accuracy', mean_pred])

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_training_1()
        {
            //# For a single-input model with 2 classes (binary classification):

            //            model = Sequential()
            //model.add(Dense(32, activation = 'relu', input_dim = 100))
            //model.add(Dense(1, activation = 'sigmoid'))
            //model.compile(optimizer = 'rmsprop',
            //              loss = 'binary_crossentropy',
            //              metrics =['accuracy'])

            //# Generate dummy data
            //import numpy as np
            //data = np.random.random((1000, 100))
            //labels = np.random.randint(2, size = (1000, 1))

            //# Train the model, iterating on the data in batches of 32 samples
            //model.fit(data, labels, epochs = 10, batch_size = 32)
            //# For a single-input model with 10 classes (categorical classification):

            //model = Sequential()
            //model.add(Dense(32, activation = 'relu', input_dim = 100))
            //model.add(Dense(10, activation = 'softmax'))
            //model.compile(optimizer = 'rmsprop',
            //              loss = 'categorical_crossentropy',
            //              metrics =['accuracy'])

            //# Generate dummy data
            //import numpy as np
            //data = np.random.random((1000, 100))
            //labels = np.random.randint(10, size = (1000, 1))

            //# Convert labels to categorical one-hot encoding
            //one_hot_labels = keras.utils.to_categorical(labels, num_classes = 10)

            //# Train the model, iterating on the data in batches of 32 samples
            //model.fit(data, one_hot_labels, epochs = 10, batch_size = 32)

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_mlp_multiclass()
        {
            //        from keras.models import Sequential
            //        from keras.layers import Dense, Dropout, Activation
            //        from keras.optimizers import SGD

            //# Generate dummy data
            //import numpy as np
            //        x_train = np.random.random((1000, 20))
            //        y_train = keras.utils.to_categorical(np.random.randint(10, size = (1000, 1)), num_classes = 10)
            //        x_test = np.random.random((100, 20))
            //        y_test = keras.utils.to_categorical(np.random.randint(10, size = (100, 1)), num_classes = 10)


            //        model = Sequential()
            //        # Dense(64) is a fully-connected layer with 64 hidden units.
            //# in the first layer, you must specify the expected input data shape:
            //# here, 20-dimensional vectors.
            //model.add(Dense(64, activation= 'relu', input_dim= 20))
            //model.add(Dropout(0.5))
            //model.add(Dense(64, activation='relu'))
            //model.add(Dropout(0.5))
            //model.add(Dense(10, activation='softmax'))

            //sgd = SGD(lr= 0.01, decay= 1e-6, momentum= 0.9, nesterov= True)
            //model.compile(loss='categorical_crossentropy',
            //              optimizer=sgd,
            //              metrics=['accuracy'])

            //model.fit(x_train, y_train,
            //          epochs=20,
            //          batch_size=128)
            //score = model.evaluate(x_test, y_test, batch_size=128)

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_mlp_binary()
        {
            //            import numpy as np
            //from keras.models import Sequential
            //from keras.layers import Dense, Dropout

            //# Generate dummy data
            //x_train = np.random.random((1000, 20))
            //y_train = np.random.randint(2, size = (1000, 1))
            //x_test = np.random.random((100, 20))
            //y_test = np.random.randint(2, size = (100, 1))

            //model = Sequential()
            //model.add(Dense(64, input_dim = 20, activation = 'relu'))
            //model.add(Dropout(0.5))
            //model.add(Dense(64, activation = 'relu'))
            //model.add(Dropout(0.5))
            //model.add(Dense(1, activation = 'sigmoid'))

            //model.compile(loss = 'binary_crossentropy',
            //              optimizer = 'rmsprop',
            //              metrics =['accuracy'])

            //model.fit(x_train, y_train,
            //          epochs = 20,
            //          batch_size = 128)
            //score = model.evaluate(x_test, y_test, batch_size = 128)

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_convnet()
        {
            //            import numpy as np
            //import keras
            //from keras.models import Sequential
            //from keras.layers import Dense, Dropout, Flatten
            //from keras.layers import Conv2D, MaxPooling2D
            //from keras.optimizers import SGD

            //# Generate dummy data
            //x_train = np.random.random((100, 100, 100, 3))
            //y_train = keras.utils.to_categorical(np.random.randint(10, size = (100, 1)), num_classes = 10)
            //x_test = np.random.random((20, 100, 100, 3))
            //y_test = keras.utils.to_categorical(np.random.randint(10, size = (20, 1)), num_classes = 10)

            //model = Sequential()
            //# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
            //# this applies 32 convolution filters of size 3x3 each.
            //model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (100, 100, 3)))
            //model.add(Conv2D(32, (3, 3), activation = 'relu'))
            //model.add(MaxPooling2D(pool_size = (2, 2)))
            //model.add(Dropout(0.25))

            //model.add(Conv2D(64, (3, 3), activation = 'relu'))
            //model.add(Conv2D(64, (3, 3), activation = 'relu'))
            //model.add(MaxPooling2D(pool_size = (2, 2)))
            //model.add(Dropout(0.25))

            //model.add(Flatten())
            //model.add(Dense(256, activation = 'relu'))
            //model.add(Dropout(0.5))
            //model.add(Dense(10, activation = 'softmax'))

            //sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
            //model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

            //model.fit(x_train, y_train, batch_size = 32, epochs = 10)
            //score = model.evaluate(x_test, y_test, batch_size = 32)

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_lstm()
        {
            //            from keras.models import Sequential
            //            from keras.layers import Dense, Dropout
            //            from keras.layers import Embedding
            //            from keras.layers import LSTM

            //            model = Sequential()
            //model.add(Embedding(max_features, output_dim = 256))
            //model.add(LSTM(128))
            //model.add(Dropout(0.5))
            //model.add(Dense(1, activation = 'sigmoid'))

            //model.compile(loss = 'binary_crossentropy',
            //              optimizer = 'rmsprop',
            //              metrics =['accuracy'])

            //model.fit(x_train, y_train, batch_size = 16, epochs = 10)
            //score = model.evaluate(x_test, y_test, batch_size = 16)

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_stacked_lstm()
        {
            //            from keras.models import Sequential
            //            from keras.layers import LSTM, Dense
            //            import numpy as np

            //data_dim = 16
            //timesteps = 8
            //num_classes = 10

            //# expected input data shape: (batch_size, timesteps, data_dim)
            //model = Sequential()
            //model.add(LSTM(32, return_sequences = True,
            //               input_shape = (timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
            //model.add(LSTM(32, return_sequences = True))  # returns a sequence of vectors of dimension 32
            //model.add(LSTM(32))  # return a single vector of dimension 32
            //model.add(Dense(10, activation = 'softmax'))

            //model.compile(loss = 'categorical_crossentropy',
            //              optimizer = 'rmsprop',
            //              metrics =['accuracy'])

            //# Generate dummy training data
            //x_train = np.random.random((1000, timesteps, data_dim))
            //y_train = np.random.random((1000, num_classes))

            //# Generate dummy validation data
            //x_val = np.random.random((100, timesteps, data_dim))
            //y_val = np.random.random((100, num_classes))

            //model.fit(x_train, y_train,
            //          batch_size = 64, epochs = 5,
            //          validation_data = (x_val, y_val))

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_stateful_stacked_lstm()
        {
            //from keras.models import Sequential
            //from keras.layers import LSTM, Dense
            //import numpy as np

            //data_dim = 16
            //timesteps = 8
            //num_classes = 10
            //batch_size = 32

            //# Expected input batch shape: (batch_size, timesteps, data_dim)
            //# Note that we have to provide the full batch_input_shape since the network is stateful.
            //# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
            //model = Sequential()
            //model.add(LSTM(32, return_sequences = True, stateful = True,
            //               batch_input_shape = (batch_size, timesteps, data_dim)))
            //model.add(LSTM(32, return_sequences = True, stateful = True))
            //model.add(LSTM(32, stateful = True))
            //model.add(Dense(10, activation = 'softmax'))

            //model.compile(loss = 'categorical_crossentropy',
            //              optimizer = 'rmsprop',
            //              metrics =['accuracy'])

            //# Generate dummy training data
            //x_train = np.random.random((batch_size * 10, timesteps, data_dim))
            //y_train = np.random.random((batch_size * 10, num_classes))

            //# Generate dummy validation data
            //x_val = np.random.random((batch_size * 3, timesteps, data_dim))
            //y_val = np.random.random((batch_size * 3, num_classes))

            //model.fit(x_train, y_train,
            //          batch_size = batch_size, epochs = 5, shuffle = False,
            //          validation_data = (x_val, y_val))

            Assert.Fail();
        }


        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
