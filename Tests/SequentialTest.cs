using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Engine.Topology;
using KerasSharp.Layers;
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

            /* Python:
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np

model = Sequential()
model.add(Dense(32, input_shape = (500,)))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'rmsprop',
      loss = 'categorical_crossentropy',
      metrics =['accuracy'])
*/
            #region doc_sequential_example_1
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 500 }));
            model.Add(new Dense(10, activation: new Softmax()));
            model.Compile(optimizer: new RootMeanSquareProp(),
                  loss: new CategoricalCrossEntropy(),
                  metrics: new Accuracy());

            #endregion

            // Assert.AreEqual(42, model.activity_regularizer);
            //Assert.AreEqual(42, model.batch_input_shape);
            //Assert.AreEqual(42, model.callback_model);
            Assert.IsTrue(model.container_nodes.SetEquals(new[] { "dense_1_ib-0", "dense_2_ib-0", "dense_1_input_ib-0" }));
            // Assert.AreEqual(42, model.dtype);
            Assert.AreEqual(true, model.built);
            Assert.AreEqual(1, model.inbound_nodes.Count);
            Assert.AreEqual(0, model.inbound_nodes[0].inbound_layers.Count);
            // Assert.AreEqual(0, model.inbound_nodes[0].input_mask.Count);
            Assert.AreEqual(1, model.inbound_nodes[0].input_masks.Count);
            Assert.AreEqual(null, model.inbound_nodes[0].input_masks[0]);
            Assert.AreEqual(1, model.inbound_nodes[0].input_shapes.Count);
            Assert.AreEqual(new int?[][] { new int?[] { null, 500 } }, model.inbound_nodes[0].input_shapes);
            Assert.AreEqual(1, model.inbound_nodes[0].input_tensors.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", model.inbound_nodes[0].input_tensors[0].ToString());
            Assert.AreEqual(0, model.inbound_nodes[0].node_indices.Count);
            Sequential l = model.inbound_nodes[0].outbound_layer as Sequential;
            Assert.IsNotNull(l);
            Assert.AreEqual(model, l);
            // Assert.AreEqual(0, model.inbound_nodes[0].output_mask);
            Assert.AreEqual(1, model.inbound_nodes[0].output_masks.Count);
            Assert.AreEqual(null, model.inbound_nodes[0].output_masks[0]);
            Assert.AreEqual(1, model.inbound_nodes[0].output_shapes.Count);
            Assert.AreEqual(new int?[] { null, 10 }, model.inbound_nodes[0].output_shapes[0]);
            Assert.AreEqual(1, model.inbound_nodes[0].output_tensors.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Softmax0_0' shape=[null, 10] dtype=Float", model.inbound_nodes[0].output_tensors[0].ToString());
            Assert.AreEqual(0, model.inbound_nodes[0].tensor_indices.Count);
            // Assert.AreEqual(42, model.input_dtype);
            Assert.AreEqual(1, model.input_layers.Count);
            Assert.IsTrue(model.input_layers[0] is InputLayer);
            //Assert.AreEqual(1, model.input_layers[0].activity_regularizer);
            Assert.AreEqual(new int?[] { null, 500 }, model.input_layers[0].batch_input_shape);
            Assert.AreEqual(true, model.input_layers[0].built);

            // --- verified until here

            Assert.AreEqual(1, model.input_layers[0].constraints);
            Assert.AreEqual(1, model.input_layers[0].dtype);
            Assert.AreEqual(1, model.input_layers[0].inbound_nodes);
            Assert.AreEqual(1, model.input_layers[0].input_dtype);
            Assert.AreEqual(1, model.input_layers[0].input_mask);
            Assert.AreEqual(1, model.input_layers[0].input_shape);
            Assert.AreEqual(1, model.input_layers[0].input_spec);
            Assert.AreEqual(1, model.input_layers[0].is_placeholder);
            Assert.AreEqual(1, model.input_layers[0].losses);
            Assert.AreEqual(1, model.input_layers[0].name);
            Assert.AreEqual(1, model.input_layers[0].non_trainable_weights);
            Assert.AreEqual(1, model.input_layers[0].outbound_nodes);
            Assert.AreEqual(1, model.input_layers[0].output);
            Assert.AreEqual(1, model.input_layers[0].output_mask);
            Assert.AreEqual(1, model.input_layers[0].output_shape);
            Assert.AreEqual(1, model.input_layers[0].stateful);
            Assert.AreEqual(1, model.input_layers[0].supports_masking);
            Assert.AreEqual(1, model.input_layers[0].trainable);
            Assert.AreEqual(1, model.input_layers[0].trainable_weights);
            Assert.AreEqual(1, model.input_layers[0].updates);
            Assert.AreEqual(1, model.input_layers[0].uses_learning_phase);
            Assert.AreEqual(1, model.input_layers[0].weights);
            Assert.AreEqual(1, model.input_layers[0]._built);
            Assert.AreEqual(1, model.input_layers[0]._constraints);
            Assert.AreEqual(1, model.input_layers[0]._flattened_layers);
            Assert.AreEqual(1, model.input_layers[0]._initial_weights);
            Assert.AreEqual(1, model.input_layers[0]._losses);
            Assert.AreEqual(1, model.input_layers[0]._non_trainable_weights);
            Assert.AreEqual(1, model.input_layers[0]._per_input_losses);
            Assert.AreEqual(1, model.input_layers[0]._per_input_updates);
            Assert.AreEqual(1, model.input_layers[0]._trainable);
            Assert.AreEqual(1, model.input_layers[0]._trainable_weights);
            Assert.AreEqual(1, model.input_layers[0]._updates);
            Assert.AreEqual(new[] { 0 }, model.input_layers_node_indices);
            Assert.AreEqual(new[] { 0 }, model.input_layers_tensor_indices);
            Assert.AreEqual(42, model.input_mask);
            Assert.AreEqual(42, model.input_names);
            Assert.AreEqual(42, model.input_shape);
            Assert.AreEqual(42, model.input_spec);
            Assert.AreEqual(42, model.is_placeholder);
            Assert.AreEqual(42, model.layers);
            Assert.AreEqual(42, model.layers_by_depth);
            Assert.AreEqual(42, model.loss);
            Assert.AreEqual(42, model.losses);
            Assert.AreEqual(42, model.loss_weights);
            Assert.AreEqual(42, model.metrics);
            Assert.AreEqual(42, model.metrics_names);
            Assert.AreEqual(42, model.metrics_tensors);
            Assert.AreEqual(42, model.model);
            Assert.AreEqual(42, model.name);
            Assert.AreEqual(42, model.nodes_by_depth);
            Assert.AreEqual(42, model.non_trainable_weights);
            Assert.AreEqual(42, model.optimizer);
            Assert.AreEqual(42, model.outbound_nodes);
            Assert.AreEqual(42, model.output);
            Assert.AreEqual(42, model.output_layers);
            Assert.AreEqual(42, model.output_layers_node_indices);
            Assert.AreEqual(42, model.output_layers_tensor_indices);
            Assert.AreEqual(42, model.output_mask);
            Assert.AreEqual(42, model.output_names);
            Assert.AreEqual(42, model.output_shape);
            Assert.AreEqual(42, model.regularizers);
            Assert.AreEqual(42, model.sample_weights);
            Assert.AreEqual(42, model.sample_weight_mode);
            Assert.AreEqual(42, model.stateful);
            Assert.AreEqual(42, model.supports_masking);
            Assert.AreEqual(42, model.targets);
            Assert.AreEqual(42, model.total_loss);
            Assert.AreEqual(42, model.Trainable);
            Assert.AreEqual(42, model.trainable);
            Assert.AreEqual(42, model.trainable_weights);
            Assert.AreEqual(42, model.updates);
            Assert.AreEqual(42, model.uses_learning_phase);
            Assert.AreEqual(42, model.weights);
            Assert.AreEqual(42, model._built);
            Assert.AreEqual(42, model._constraints);
            Assert.AreEqual(42, model._feed_inputs);
            Assert.AreEqual(42, model._feed_input_names);
            Assert.AreEqual(42, model._feed_input_shapes);
            Assert.AreEqual(42, model._feed_sample_weight_modes);
            Assert.AreEqual(42, model._flattened_layers);
            Assert.AreEqual(42, model._initial_weights);
            Assert.AreEqual(42, model._losses);
            Assert.AreEqual(42, model._non_trainable_weights);
            Assert.AreEqual(42, model._output_mask_cache);
            Assert.AreEqual(42, model._output_shape_cache);
            Assert.AreEqual(42, model._output_tensor_cache);
            Assert.AreEqual(42, model._per_input_losses);
            Assert.AreEqual(42, model._per_input_updates);
            Assert.AreEqual(42, model._trainable);
            Assert.AreEqual(42, model._trainable_weights);
            Assert.AreEqual(42, model._updates);
        }

        [Test]
        public void sequential_guide_1()
        {
            var model = new Sequential(new List<Layer> {
                new Dense(32, input_shape: new int?[] { 784 }),
                new Activation("relu"),
                new Dense(10),
                new Activation("softmax"),
            });

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_2()
        {
            var model = new Sequential();
            model.Add(new Dense(32, input_dim: 784));
            model.Add(new Activation("relu"));

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_3()
        {
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 784 }));
            model = new Sequential();
            model.Add(new Dense(32, input_dim: 784));

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_compilation_1()
        {
            var K = KerasSharp.Backends.Current.K;

            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 784 }));

            // For a multi-class classification problem
            model.Compile(optimizer: "rmsprop",
                          loss: "categorical_crossentropy",
                          metrics: new[] { "accuracy" });

            // For a binary classification problem
            model.Compile(optimizer: "rmsprop",
                          loss: "binary_crossentropy",
                          metrics: new[] { "accuracy" });

            // For a mean squared error regression problem
            model.Compile(optimizer: "rmsprop",
                          loss: "mse");

            // For custom metrics
            Func<Tensor, Tensor, Tensor> mean_pred = (Tensor y_true, Tensor y_pred) =>
            {
                return K.mean(y_pred);
            };

            model.Compile(optimizer: "rmsprop",
                          loss: "binary_crossentropy",
                          metrics: new object[] { "accuracy", mean_pred });

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_training_1()
        {
            // For a single-input model with 2 classes (binary classification):

            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_dim: 100));
            model.Add(new Dense(1, activation: "sigmoid"));
            model.Compile(optimizer: "rmsprop",
                          loss: "binary_crossentropy",
                          metrics: new[] { "accuracy" });

            // Generate dummy data
            double[,] data = Accord.Math.Matrix.Random(1000, 100);
            int[] labels = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            // Train the model, iterating on the data in batches of 32 samples
            model.fit(data, labels, epochs: 10, batch_size: 32);
            // For a single-input model with 10 classes (categorical classification):

            model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_dim: 100));
            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(optimizer: "rmsprop",
                          loss: "categorical_crossentropy",
                          metrics: new[] { "accuracy" });

            // Generate dummy data
            data = Accord.Math.Matrix.Random(1000, 100);
            labels = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            // Convert labels to categorical one-hot encoding
            double[,] one_hot_labels = Accord.Math.Matrix.OneHot(labels, columns: 10);

            // Train the model, iterating on the data in batches of 32 samples
            model.fit(data, one_hot_labels, epochs: 10, batch_size: 32);

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_mlp_multiclass()
        {
            // Generate dummy data
            double[,] x_train = Accord.Math.Matrix.Random(1000, 20);
            int[] y_train = Accord.Math.Vector.Random(1000, min: 0, max: 10);
            double[,] x_test = Accord.Math.Matrix.Random(1000, 20);
            int[] y_test = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            var model = new Sequential();
            // Dense(64) is a fully-connected layer with 64 hidden units.
            // in the first layer, you must specify the expected input data shape:
            // here, 20-dimensional vectors.

            model.Add(new Dense(64, activation: "relu", input_dim: 20));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(10, activation: "softmax"));

            var sgd = new SGD(lr: 0.01, decay: 1e-6, momentum: 0.9, nesterov: true);
            model.Compile(loss: "categorical_crossentropy",
                          optimizer: sgd,
                          metrics: new[] { "accuracy" });

            model.fit(x_train, y_train,
                      epochs: 20,
                      batch_size: 128);

            var score = model.evaluate(x_test, y_test, batch_size: 128);

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_mlp_binary()
        {
            // Generate dummy data
            double[,] x_train = Accord.Math.Matrix.Random(1000, 20);
            int[] y_train = Accord.Math.Vector.Random(1000, min: 0, max: 10);
            double[,] x_test = Accord.Math.Matrix.Random(1000, 20);
            int[] y_test = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            var model = new Sequential();
            model.Add(new Dense(64, input_dim: 20, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(loss: "binary_crossentropy",
                          optimizer: "rmsprop",
                          metrics: new[] { "accuracy" });

            model.fit(x_train, y_train,
                      epochs: 20,
                      batch_size: 128);

            var score = model.evaluate(x_test, y_test, batch_size: 128);

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_convnet()
        {
            // Generate dummy data
            double[,,,] x_train = (double[,,,])Accord.Math.Matrix.Create(typeof(double), new int[] { 100, 100, 100, 3 }); // TODO: Add a better overload in Accord
            int[] y_train = Accord.Math.Vector.Random(100, min: 0, max: 10);
            double[,,,] x_test = (double[,,,])Accord.Math.Matrix.Create(typeof(double), new int[] { 20, 100, 100, 3 }); // TODO: Add a better overload in Accord
            int[] y_test = Accord.Math.Vector.Random(100, min: 0, max: 10);

            var model = new Sequential();
            // input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
            // this applies 32 convolution filters of size 3x3 each.
            model.Add(new Conv2D(32, new[] { 3, 3 }, activation: "relu", input_shape: new int?[] { 100, 100, 3 }));
            model.Add(new Conv2D(32, new[] { 3, 3 }, activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: new[] { 2, 2 }));
            model.Add(new Dropout(0.25));

            model.Add(new Conv2D(64, new[] { 3, 3 }, activation: "relu"));
            model.Add(new Conv2D(64, new[] { 3, 3 }, activation: "relu"));
            model.Add(new MaxPooling2D(pool_size: new[] { 2, 2 }));
            model.Add(new Dropout(0.25));

            model.Add(new Flatten());
            model.Add(new Dense(256, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(10, activation: "softmax"));

            var sgd = new SGD(lr: 0.01, decay: 1e-6, momentum: 0.9, nesterov: true);
            model.Compile(loss: "categorical_crossentropy", optimizer: sgd);

            model.fit(x_train, y_train, batch_size: 32, epochs: 10);
            var score = model.evaluate(x_test, y_test, batch_size: 32);

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_lstm()
        {
            // Generate dummy data
            double[][][] x_train = null; // TODO: Generate 100 random sequences, with random lengths
            int[] y_train = Accord.Math.Vector.Random(100, min: 0, max: 10);
            double[][][] x_test = null; // TODO: Generate 50 random sequences, with random lengths
            int[] y_test = Accord.Math.Vector.Random(50, min: 0, max: 10);

            int max_features = 1024;

            var model = new Sequential();
            model.Add(new Embedding(max_features, output_dim: 256));
            model.Add(new LSTM(128));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(1, activation: "sigmoid"));

            model.Compile(loss: "binary_crossentropy",
                          optimizer: "rmsprop",
                          metrics: new[] { "accuracy" });

            model.fit(x_train, y_train, batch_size: 16, epochs: 10);
            var score = model.evaluate(x_test, y_test, batch_size: 16);

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_stacked_lstm()
        {
            int data_dim = 16;
            int timesteps = 8;
            int num_classes = 10;

            // expected input data shape: (batch_size, timesteps, data_dim)
            var model = new Sequential();
            model.Add(new LSTM(32, return_sequences: true,
                           input_shape: new[] { timesteps, data_dim })); // returns a sequence of vectors of dimension 32
            model.Add(new LSTM(32, return_sequences: true));         // returns a sequence of vectors of dimension 32
            model.Add(new LSTM(32));                                  // return a single vector of dimension 32
            model.Add(new Dense(10, activation: "softmax"));

            model.Compile(loss: "categorical_crossentropy",
                          optimizer: "rmsprop",
                          metrics: new[] { "accuracy" });

            // Generate dummy training data
            double[][][] x_train = null; // Accord.Math.Jagged.Random(1000, timesteps, data_dim); // TODO: Add better method in Accord
            int[] y_train = Accord.Math.Vector.Random(1000, min: 0, max: num_classes);

            // Generate dummy validation data
            double[,,] x_val = null; // Accord.Math.Jagged.Random(1000, timesteps, data_dim); // TODO: Add better method in Accord
            int[] y_val = Accord.Math.Vector.Random(1000, min: 0, max: num_classes);

            model.fit(x_train, y_train,
                      batch_size: 64, epochs: 5,
                      validation_data: new object[] { x_val, y_val });

            Assert.Fail();
        }

        [Test]
        public void sequential_guide_stateful_stacked_lstm()
        {
            int data_dim = 16;
            int timesteps = 8;
            int num_classes = 10;
            int batch_size = 32;

            // Expected input batch shape: (batch_size, timesteps, data_dim)
            // Note that we have to provide the full batch_input_shape since the network is stateful.
            // the sample of index i in batch k is the follow-up for the sample i in batch k-1.
            var model = new Sequential();
            model.Add(new LSTM(32, return_sequences: true, stateful: true,
                           batch_input_shape: new int?[] { batch_size, timesteps, data_dim }));
            model.Add(new LSTM(32, return_sequences: true, stateful: true));
            model.Add(new LSTM(32, stateful: true));
            model.Add(new Dense(10, activation: "softmax"));

            model.Compile(loss: "categorical_crossentropy",
                          optimizer: "rmsprop",
                          metrics: new[] { "accuracy" });

            // Generate dummy training data
            double[][][] x_train = null; // Accord.Math.Jagged.Random(1000, timesteps, data_dim); // TODO: Add better method in Accord
            int[] y_train = Accord.Math.Vector.Random(1000, min: 0, max: num_classes);

            // Generate dummy validation data
            double[,,] x_val = null; // Accord.Math.Jagged.Random(1000, timesteps, data_dim); // TODO: Add better method in Accord
            int[] y_val = Accord.Math.Vector.Random(1000, min: 0, max: num_classes);

            model.fit(x_train, y_train,
                      batch_size: batch_size, epochs: 5, shuffle: false,
                      validation_data: new object[] { x_val, y_val });

            Assert.Fail();
        }


        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
