using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Engine.Topology;
using KerasSharp.Initializers;
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

using static KerasSharp.Python;

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
            model.Compile(optimizer: new RMSProp(),
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
            string str = model.inbound_nodes[0].ToString();
            Assert.AreEqual("{  } => sequential_1 ([[null, 500]] -> [[null, 10]])", str);
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
            (Layer historyLayer, int historyNodeIndex, int historyTensorIndex) = model.inbound_nodes[0].output_tensors[0]._keras_history.Value;
            Assert.AreEqual("dense_2 ([[null, 32]] -> [[null, 10]])", historyLayer.ToString());
            Assert.AreEqual(0, historyNodeIndex);
            Assert.AreEqual(0, historyTensorIndex);
            Assert.AreEqual(tuple(null, 10), model.inbound_nodes[0].output_tensors[0]._keras_shape);
            Assert.AreEqual(false, model.inbound_nodes[0].output_tensors[0]._uses_learning_phase);
            Assert.AreEqual(0, model.inbound_nodes[0].tensor_indices.Count);
            // Assert.AreEqual(42, model.input_dtype);


            Assert.AreEqual(1, model.input_layers.Count);
            Assert.IsTrue(model.input_layers[0] is InputLayer);
            //Assert.AreEqual(1, model.input_layers[0].activity_regularizer);
            Assert.AreEqual(new int?[] { null, 500 }, model.input_layers[0].batch_input_shape);
            Assert.AreEqual(true, model.input_layers[0].built);
            //Assert.AreEqual(1, model.input_layers[0].constraints);
            Assert.AreEqual(DataType.Float, model.input_layers[0].dtype);
            Assert.AreEqual(1, model.input_layers[0].inbound_nodes.Count);
            Assert.AreEqual(1, model.input_layers[0].inbound_nodes.Count);
            var layer_node = model.input_layers[0].inbound_nodes[0];

            str = layer_node.ToString();
            Assert.AreEqual("{  } => dense_1_input ([[null, 500]] -> [[null, 500]])", str);
            Assert.AreEqual(0, layer_node.inbound_layers.Count);
            Assert.AreEqual(1, layer_node.input_masks.Count);
            Assert.AreEqual(null, layer_node.input_masks[0]);
            Assert.AreEqual(1, layer_node.input_shapes.Count);
            Assert.AreEqual(new int?[][] { new int?[] { null, 500 } }, layer_node.input_shapes);
            Assert.AreEqual(1, layer_node.input_tensors.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", layer_node.input_tensors[0].ToString());
            (historyLayer, historyNodeIndex, historyTensorIndex) = layer_node.input_tensors[0]._keras_history.Value;
            Assert.AreEqual("dense_1_input ([[null, 500]] -> [[null, 500]])", historyLayer.ToString());
            Assert.AreEqual(0, historyNodeIndex);
            Assert.AreEqual(0, historyTensorIndex);
            Assert.AreEqual(tuple(null, 500), layer_node.input_tensors[0]._keras_shape);
            Assert.AreEqual(false, layer_node.input_tensors[0]._uses_learning_phase);
            Assert.AreEqual(0, layer_node.node_indices.Count);
            Assert.IsTrue(layer_node.outbound_layer is InputLayer);
            Assert.AreEqual(model.input_layers[0], layer_node.outbound_layer);
            Assert.AreEqual(1, layer_node.output_masks.Count);
            Assert.AreEqual(null, layer_node.output_masks[0]);
            Assert.AreEqual(1, layer_node.output_shapes.Count);
            Assert.AreEqual(new int?[] { null, 500 }, layer_node.output_shapes[0]);
            Assert.AreEqual(1, layer_node.output_tensors.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", layer_node.output_tensors[0].ToString());

            (historyLayer, historyNodeIndex, historyTensorIndex) = layer_node.output_tensors[0]._keras_history.Value;
            str = historyLayer.ToString();
            Assert.AreEqual("dense_1_input ([[null, 500]] -> [[null, 500]])", str);
            Assert.AreEqual(0, historyNodeIndex);
            Assert.AreEqual(0, historyTensorIndex);

            Assert.AreEqual(tuple(null, 10), model.inbound_nodes[0].output_tensors[0]._keras_shape);

            Assert.AreEqual(tuple(null, 500), layer_node.output_tensors[0]._keras_shape);
            Assert.AreEqual(false, layer_node.output_tensors[0]._uses_learning_phase);
            Assert.AreEqual(0, layer_node.tensor_indices.Count);

            //model.input_layers[0].__dict__:
            //{
            // '_built': True,
            // '_initial_weights': None,
            // '_losses': [],
            // '_non_trainable_weights': [],
            // '_per_input_losses': {},
            // '_per_input_updates': {},
            // '_trainable_weights': [],
            // '_updates': [],
            // 'batch_input_shape': (None, 500),
            // 'dtype': 'float32',
            // 'inbound_nodes': [<keras.engine.topology.Node at 0x7f62077b8710>],
            // 'input_spec': None,
            // 'is_placeholder': True,
            // 'name': 'dense_1_input',
            // 'outbound_nodes': [<keras.engine.topology.Node at 0x7f61f96eff28>],
            // 'sparse': False,
            // 'supports_masking': False,
            // 'trainable': False
            // }

            var inputLayer = model.input_layers[0] as InputLayer;
            str = inputLayer.ToString();
            Assert.AreEqual("dense_1_input ([[null, 500]] -> [[null, 500]])", str);
            Assert.AreEqual(null, inputLayer.input_dtype); // legacy
            Assert.AreEqual(null, inputLayer.input_mask);
            Assert.AreEqual(1, inputLayer.input_shape.Count);
            Assert.AreEqual(tuple(null, 500), inputLayer.input_shape[0]);
            Assert.AreEqual(null, inputLayer.input_spec);
            Assert.AreEqual(true, inputLayer.is_placeholder);
            Assert.AreEqual(0, inputLayer.losses.Count);
            Assert.AreEqual("dense_1_input", inputLayer.name);
            Assert.AreEqual(0, inputLayer.non_trainable_weights.Count);
            Assert.AreEqual(1, inputLayer.outbound_nodes.Count);

            str = inputLayer.outbound_nodes[0].ToString();
            Assert.AreEqual("{ dense_1_input ([[null, 500]] -> [[null, 500]]) } => dense_1 ([[null, 500]] -> [[null, 32]])", str);

            Assert.AreEqual(1, inputLayer.output.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", inputLayer.output[0].ToString());
            (historyLayer, historyNodeIndex, historyTensorIndex) = inputLayer.output[0]._keras_history.Value;
            Assert.AreEqual("dense_1_input ([[null, 500]] -> [[null, 500]])", historyLayer.ToString());
            Assert.AreEqual(0, historyNodeIndex);
            Assert.AreEqual(0, historyTensorIndex);

            Assert.AreEqual(tuple(null, 500), inputLayer.output[0]._keras_shape);
            Assert.AreEqual(false, inputLayer.output[0]._uses_learning_phase);

            Assert.AreEqual(null, inputLayer.output_mask);
            Assert.AreEqual(1, inputLayer.output_shape.Count);
            Assert.AreEqual(tuple(null, 500), inputLayer.output_shape[0]);
            Assert.AreEqual(false, inputLayer.stateful);
            Assert.AreEqual(false, inputLayer.supports_masking);
            Assert.AreEqual(0, inputLayer.trainable_weights.Count);
            Assert.AreEqual(0, inputLayer.updates.Count);
            Assert.AreEqual(false, inputLayer.uses_learning_phase);
            Assert.AreEqual(0, inputLayer.weights.Count);
            Assert.AreEqual(true, inputLayer._built);
            //Assert.AreEqual(1, inputLayer._constraints);
            Assert.AreEqual(0, inputLayer._losses.Count);
            Assert.AreEqual(0, inputLayer._non_trainable_weights.Count);
            Assert.AreEqual(0, inputLayer._per_input_losses.Count);
            Assert.AreEqual(0, inputLayer._per_input_updates.Count);
            Assert.AreEqual(0, inputLayer._trainable_weights.Count);
            Assert.AreEqual(0, inputLayer._updates.Count);


            Assert.AreEqual(new[] { 0 }, model.input_layers_node_indices);
            Assert.AreEqual(new[] { 0 }, model.input_layers_tensor_indices);
            Assert.AreEqual(null, model.input_mask);
            Assert.AreEqual(1, model.input_names.Count);
            Assert.AreEqual("dense_1_input", model.input_names[0]);
            Assert.AreEqual(tuple(null, 500), model.input_shape[0]);
            Assert.AreEqual(2, model.layers.Count);
            Assert.AreEqual("dense_1 ([[null, 500]] -> [[null, 32]])", model.layers[0].ToString());
            Assert.AreEqual("dense_2 ([[null, 32]] -> [[null, 10]])", model.layers[1].ToString());
            Assert.AreEqual(true, model.loss["__K__single__"] is CategoricalCrossEntropy);
            Assert.AreEqual(0, model.losses.Count);
            Assert.AreEqual(null, model.loss_weights);
            Assert.AreEqual(true, model.metrics["__K__single__"][0] is Accuracy);


            Assert.AreEqual("sequential_1", model.name);
            Assert.AreEqual(3, model.nodes_by_depth.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Add1_0' shape=[null, 32] dtype=Float", model.nodes_by_depth[0][0].input_tensors[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Softmax0_0' shape=[null, 10] dtype=Float", model.nodes_by_depth[0][0].output_tensors[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", model.nodes_by_depth[1][0].input_tensors[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Add1_0' shape=[null, 32] dtype=Float", model.nodes_by_depth[1][0].output_tensors[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", model.nodes_by_depth[2][0].input_tensors[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", model.nodes_by_depth[2][0].output_tensors[0].ToString());

            Assert.AreEqual(0, model.non_trainable_weights.Count);
            Assert.AreEqual(4, model.trainable_weights.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Add0_0' shape=[500, 32] dtype=Float", model.trainable_weights[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Const3_0' shape=[] dtype=Float", model.trainable_weights[1].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Add0_0' shape=[32, 10] dtype=Float", model.trainable_weights[2].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Const3_0' shape=[] dtype=Float", model.trainable_weights[3].ToString());
            Assert.AreEqual(true, model.optimizer is RMSProp);
            Assert.AreEqual(0, model.outbound_nodes.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Softmax0_0' shape=[null, 10] dtype=Float", model.output[0].ToString());
            Assert.AreEqual(1, model.output_layers.Count);

            var outputLayer = model.output_layers[0] as Dense;
            Assert.AreEqual(null, outputLayer.activity_regularizer);
            //Assert.AreEqual(0, outputLayer.batch_input_shape);
            Assert.AreEqual(true, outputLayer.built);
            Assert.AreEqual(0, outputLayer.constraints.Count);
            Assert.AreEqual(DataType.Float, outputLayer.dtype);
            Assert.AreEqual(1, outputLayer.inbound_nodes.Count);
            Assert.AreEqual("{ dense_1 ([[null, 500]] -> [[null, 32]]) } => dense_2 ([[null, 32]] -> [[null, 10]])", outputLayer.inbound_nodes[0].ToString());
            Assert.AreEqual(1, outputLayer.input.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Add1_0' shape=[null, 32] dtype=Float", outputLayer.input[0].ToString());
            // Assert.AreEqual(0, outputLayer.input_dtype);
            Assert.AreEqual(null, outputLayer.input_mask);
            Assert.AreEqual(1, outputLayer.input_shape.Count);
            Assert.AreEqual(tuple(null, 32), outputLayer.input_shape[0]);
            Assert.AreEqual(1, outputLayer.input_spec.Count);
            Assert.AreEqual("dtype=Float, shape=null, ndim=, max_ndim=, min_ndim=2, axes=[[-1, 32]]", outputLayer.input_spec[0].ToString());
            // Assert.AreEqual(0, outputLayer.is_placeholder);
            Assert.AreEqual(0, outputLayer.losses.Count);
            Assert.AreEqual("dense_2", outputLayer.name);
            Assert.AreEqual(0, outputLayer.non_trainable_weights.Count);
            Assert.AreEqual(0, outputLayer.outbound_nodes.Count);
            Assert.AreEqual(1, outputLayer.output.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Softmax0_0' shape=[null, 10] dtype=Float", outputLayer.output[0].ToString());
            Assert.AreEqual(null, outputLayer.output_mask);
            Assert.AreEqual(1, outputLayer.output_shape.Count);
            Assert.AreEqual(tuple(null, 10), outputLayer.output_shape[0]);
            Assert.AreEqual(false, outputLayer.stateful);
            Assert.AreEqual(true, outputLayer.supports_masking);
            Assert.AreEqual(true, outputLayer.trainable);
            Assert.AreEqual(2, outputLayer.trainable_weights.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Add0_0' shape=[32, 10] dtype=Float", outputLayer.trainable_weights[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Const3_0' shape=[] dtype=Float", outputLayer.trainable_weights[1].ToString());
            Assert.AreEqual(0, outputLayer.updates.Count);
            //Assert.AreEqual(0, outputLayer.uses_learning_phase);
            Assert.AreEqual(2, outputLayer.weights.Count);
            Assert.AreEqual(true, outputLayer._built);
            Assert.AreEqual(0, outputLayer._constraints.Count);
            //Assert.AreEqual(null, outputLayer._flattened_layers);
            Assert.AreEqual(null, outputLayer._initial_weights);
            Assert.AreEqual(0, outputLayer._losses.Count);
            Assert.AreEqual(0, outputLayer._non_trainable_weights.Count);
            Assert.AreEqual(0, outputLayer._per_input_losses.Count);
            Assert.AreEqual(0, outputLayer._per_input_updates.Count);

            Assert.AreEqual(true, outputLayer.kernel_initializer is GlorotUniform);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Add0_0' shape=[32, 10] dtype=Float", outputLayer.kernel.ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Const3_0' shape=[] dtype=Float", outputLayer.bias.ToString());
            Assert.AreEqual(10, outputLayer.units);
            Assert.AreEqual(true, outputLayer.activation is Softmax);
            Assert.AreEqual(true, outputLayer.use_bias);
            Assert.AreEqual(true, outputLayer.bias_initializer is Zeros);
            Assert.AreEqual(null, outputLayer.activity_regularizer);
            Assert.AreEqual(true, outputLayer._trainable);
            Assert.AreEqual(2, outputLayer._trainable_weights.Count);
            Assert.AreEqual(0, outputLayer._updates.Count);


            Assert.AreEqual(1, model.output_layers_node_indices.Count);
            Assert.AreEqual(0, model.output_layers_node_indices[0]);
            Assert.AreEqual(1, model.output_layers_tensor_indices.Count);
            Assert.AreEqual(0, model.output_layers_tensor_indices[0]);
            Assert.AreEqual(null, model.output_mask);
            Assert.AreEqual("dense_2", model.output_names[0]);
            Assert.AreEqual(tuple(null, 10), model.output_shape[0]);
            Assert.AreEqual(null, model.regularizers);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2_sample_weights_0' shape=[null] dtype=Float", model.sample_weights[0].ToString());
            Assert.AreEqual(null, model.sample_weight_mode);
            Assert.AreEqual(false, model.stateful);
            Assert.AreEqual(false, model.supports_masking);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2_target_0' shape=[null, null] dtype=Float", model.targets[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'loss/Mul0_0' shape=[] dtype=Float", model.total_loss.ToString());
            Assert.AreEqual(true, model.trainable);
            Assert.AreEqual(4, model.trainable_weights.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Add0_0' shape=[500, 32] dtype=Float", model.trainable_weights[0].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1/Const3_0' shape=[] dtype=Float", model.trainable_weights[1].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Add0_0' shape=[32, 10] dtype=Float", model.trainable_weights[2].ToString());
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_2/Const3_0' shape=[] dtype=Float", model.trainable_weights[3].ToString());
            Assert.AreEqual(0, model.updates.Count);
            Assert.AreEqual(false, model.uses_learning_phase);
            Assert.AreEqual(model.trainable_weights, model.weights);
            Assert.AreEqual(true, model._built);
            Assert.AreEqual(0, model._constraints.Count);
            Assert.AreEqual(1, model._feed_inputs.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'dense_1_input_0' shape=[null, 500] dtype=Float", model._feed_inputs[0].ToString());
            Assert.AreEqual("dense_1_input", model._feed_input_names[0]);
            Assert.AreEqual(null, model._feed_input_shapes);
            Assert.AreEqual(null, model._feed_sample_weight_modes);
            Assert.AreEqual(2, model._flattened_layers.Count);
            Assert.AreEqual("dense_1 ([[null, 500]] -> [[null, 32]])", model._flattened_layers[0].ToString());
            Assert.AreEqual("dense_2 ([[null, 32]] -> [[null, 10]])", model._flattened_layers[1].ToString());
            Assert.AreEqual(null, model._initial_weights);
            Assert.AreEqual(0, model._losses.Count);
            Assert.AreEqual(0, model._non_trainable_weights.Count);
            Assert.AreEqual(1, model._output_mask_cache.Count);
            Assert.AreEqual(null, model._output_mask_cache["1_0"][0]);
            Assert.AreEqual(0, model._output_shape_cache.Count);
            Assert.AreEqual(0, model._output_tensor_cache.Count);
            //Assert.AreEqual(0, model._per_input_losses.Count);
            //Assert.AreEqual(0, model._per_input_updates.Count);
            Assert.AreEqual(true, model._trainable);
            //Assert.AreEqual(42, model._trainable_weights.Count);
            //Assert.AreEqual(42, model._updates.Count);

            // --- verified until here ---

            Assert.AreEqual(1, model.metrics_tensors.Count);
            Assert.AreEqual("KerasSharp.Engine.Topology.Tensor 'Mean0_0' shape=[] dtype=Float", model.metrics_tensors[0].ToString());

            Assert.AreEqual(2, model.metrics_names.Count);
            Assert.AreEqual("loss", model.metrics_names[0]);
            Assert.AreEqual("acc", model.metrics_names[1]);

            Assert.AreEqual(model.activity_regularizer, model.model.activity_regularizer);
            Assert.AreEqual(model.batch_input_shape, model.model.batch_input_shape);
            Assert.AreEqual(model.built, model.model.built);
            Assert.AreEqual(model.constraints, model.model.constraints);
            Assert.AreEqual(model.container_nodes, model.model.container_nodes);
            Assert.AreEqual(model.dtype, model.model.dtype);
            Assert.AreEqual(model.input, model.model.input);
            Assert.AreEqual(model.input_dtype, model.model.input_dtype);
            Assert.AreEqual(model.input_layers, model.model.input_layers);
            Assert.AreEqual(model.input_layers_node_indices, model.model.input_layers_node_indices);
            Assert.AreEqual(model.input_layers_tensor_indices, model.model.input_layers_tensor_indices);
            Assert.AreEqual(model.input_mask, model.model.input_mask);
            Assert.AreEqual(model.input_names, model.model.input_names);
            Assert.AreEqual(model.input_shape, model.model.input_shape);
            Assert.AreEqual(model.input_spec, model.model.input_spec);
            Assert.AreEqual(model.is_placeholder, model.model.is_placeholder);
            Assert.AreEqual(model.loss, model.model.loss);
            Assert.AreEqual(model.losses, model.model.losses);
            Assert.AreEqual(model.loss_weights, model.model.loss_weights);
            Assert.AreEqual(model.metrics, model.model.metrics);
            Assert.AreEqual(model.metrics_names, model.model.metrics_names);
            Assert.AreEqual(model.metrics_tensors, model.model.metrics_tensors);
            Assert.AreEqual(model.nodes_by_depth, model.model.nodes_by_depth);
            Assert.AreEqual(model.non_trainable_weights, model.model.non_trainable_weights);
            Assert.AreEqual(model.optimizer, model.model.optimizer);
            Assert.AreEqual(model.outbound_nodes, model.model.outbound_nodes);
            Assert.AreEqual(model.output, model.model.output);
            Assert.AreEqual(model.output_layers, model.model.output_layers);
            Assert.AreEqual(model.output_layers_node_indices, model.model.output_layers_node_indices);
            Assert.AreEqual(model.output_layers_tensor_indices, model.model.output_layers_tensor_indices);
            Assert.AreEqual(model.output_mask, model.model.output_mask);
            Assert.AreEqual(model.output_names, model.model.output_names);
            Assert.AreEqual(model.output_shape, model.model.output_shape);
            Assert.AreEqual(model.regularizers, model.model.regularizers);
            Assert.AreEqual(model.sample_weights, model.model.sample_weights);
            Assert.AreEqual(model.sample_weight_mode, model.model.sample_weight_mode);
            Assert.AreEqual(model.stateful, model.model.stateful);
            Assert.AreEqual(model.supports_masking, model.model.supports_masking);
            Assert.AreEqual(model.targets, model.model.targets);
            Assert.AreEqual(model.total_loss, model.model.total_loss);
            Assert.AreEqual(model.trainable, model.model.trainable);
            Assert.AreEqual(model.trainable_weights, model.model.trainable_weights);
            Assert.AreEqual(model.updates, model.model.updates);
            Assert.AreEqual(model.uses_learning_phase, model.model.uses_learning_phase);
            Assert.AreEqual(model.weights, model.model.weights);
            Assert.AreEqual(model._built, model.model._built);
            Assert.AreEqual(model._constraints, model.model._constraints);
            Assert.AreEqual(model._feed_inputs, model.model._feed_inputs);
            Assert.AreEqual(model._feed_input_names, model.model._feed_input_names);
            Assert.AreEqual(null, model._feed_sample_weight_modes);
            Assert.AreEqual(1, model.model._feed_sample_weight_modes.Count);
            Assert.AreEqual(null, model.model._feed_sample_weight_modes[0]);
            Assert.AreEqual(model._initial_weights, model.model._initial_weights);
            Assert.AreEqual(model._losses, model.model._losses);
            Assert.AreEqual(model._non_trainable_weights, model.model._non_trainable_weights);
            Assert.AreEqual(model._output_mask_cache, model.model._output_mask_cache);
            Assert.AreEqual(model._output_shape_cache, model.model._output_shape_cache);
            Assert.AreEqual(model._output_tensor_cache, model.model._output_tensor_cache);
            Assert.AreEqual(model._per_input_losses, model.model._per_input_losses);
            Assert.AreEqual(model._per_input_updates, model.model._per_input_updates);
            Assert.AreEqual(model._trainable, model.model._trainable);
            Assert.AreEqual(model._updates, model.model._updates);


            Assert.AreEqual("sequential_1_model", model.model.name);

            // The following assertions might need to be changed as development progresses:
            Assert.AreNotEqual(model._trainable_weights, model.model._trainable_weights);
            Assert.AreNotEqual(model._flattened_layers, model.model._flattened_layers);
            Assert.AreNotEqual(model._feed_input_shapes, model.model._feed_input_shapes);
            Assert.AreNotEqual(model.layers_by_depth, model.model.layers_by_depth);
            Assert.AreNotEqual(model.layers, model.model.layers);
            Assert.AreNotEqual(model.inbound_nodes, model.model.inbound_nodes);
            Assert.AreNotEqual(model.callback_model, model.model.callback_model);
        }

        [TestCase(Setup.TENSORFLOW)]
        [TestCase(Setup.CNTK)]
        public void sequential_guide_1(string backend)
        {
            KerasSharp.Backends.Current.Switch(backend);
            /*
                        model = Sequential([
                            Dense(32, input_shape=(784,)),
                            Activation('relu'),
                            Dense(10),
                            Activation('softmax'),
                        ])
            */
            var model = new Sequential(new List<Layer> {
                new Dense(32, input_shape: new int?[] { 784 }),
                new Activation("relu"),
                new Dense(10),
                new Activation("softmax"),
            });
        }

        [Test]
        public void sequential_guide_2()
        {
            var model = new Sequential();
            model.Add(new Dense(32, input_dim: 784));
            model.Add(new Activation("relu"));
        }

        [Test]
        public void sequential_guide_3()
        {
            var model = new Sequential();
            model.Add(new Dense(32, input_shape: new int?[] { 784 }));
            model = new Sequential();
            model.Add(new Dense(32, input_dim: 784));
        }

        [Test]
        public void sequential_guide_compilation_1()
        {
            var K = KerasSharp.Backends.Current.K;

            {
                var model = new Sequential();
                model.Add(new Dense(32, input_shape: new int?[] { 784 }));

                // For a multi-class classification problem
                model.Compile(optimizer: "rmsprop",
                              loss: "categorical_crossentropy",
                              metrics: new[] { "accuracy" });
                K.clear_session();
            }

            {
                var model = new Sequential();
                model.Add(new Dense(32, input_shape: new int?[] { 784 }));

                // For a binary classification problem
                model.Compile(optimizer: "rmsprop",
                              loss: "binary_crossentropy",
                              metrics: new[] { "accuracy" });
                K.clear_session();
            }

            {
                var model = new Sequential();
                model.Add(new Dense(32, input_shape: new int?[] { 784 }));

                // For a mean squared error regression problem
                model.Compile(optimizer: "rmsprop",
                          loss: "mse");
                K.clear_session();
            }

            {
                var model = new Sequential();
                model.Add(new Dense(32, input_shape: new int?[] { 784 }));

                // For custom metrics
                Func<Tensor, Tensor, Tensor> mean_pred = (Tensor y_true, Tensor y_pred) =>
                {
                    return K.mean(y_pred);
                };

                model.Compile(optimizer: "rmsprop",
                              loss: "binary_crossentropy",
                              metrics: new object[] { "accuracy", mean_pred });
                K.clear_session();
            }
        }

        [TestCase(Setup.TENSORFLOW)]
        [TestCase(Setup.CNTK)]
        public void sequential_guide_training_1(string backend)
        {
            KerasSharp.Backends.Current.Switch(backend);

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
        }

        [Test]
        public void sequential_guide_training_2()
        {
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_dim: 100));
            model.Add(new Dense(10, activation: "softmax"));
            model.Compile(optimizer: "rmsprop",
                          loss: "categorical_crossentropy",
                          metrics: new[] { "accuracy" });

            // Generate dummy data
            double[][] data = Accord.Math.Jagged.Random(1000, 100);
            int[] labels = Accord.Math.Vector.Random(1000, min: 0, max: 10);

            // Convert labels to categorical one-hot encoding
            double[][] one_hot_labels = Accord.Math.Jagged.OneHot(labels, columns: 10);

            // Train the model, iterating on the data in batches of 32 samples
            model.fit(data, one_hot_labels, epochs: 10, batch_size: 32);
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
        }

        [Test]
        public void sequential_guide_convnet()
        {
            // Generate dummy data
            double[,,,] x_train = (double[,,,])Accord.Math.Matrix.Zeros<double>(new int[] { 100, 100, 100, 3 }); // TODO: Add a better overload in Accord
            int[] y_train = Accord.Math.Vector.Random(100, min: 0, max: 10);
            double[,,,] x_test = (double[,,,])Accord.Math.Matrix.Zeros<double>(new int[] { 20, 100, 100, 3 }); // TODO: Add a better overload in Accord
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
                      validation_data: new Array[] { x_val, y_val });
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
                      batch_size: batch_size, epochs: 5, shuffle: Shuffle.False,
                      validation_data: new Array[] { x_val, y_val });
        }


        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
