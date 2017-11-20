// Keras-Sharp: C# port of the Keras library
// https://github.com/cesarsouza/keras-sharp
//
// Based under the Keras library for Python. See LICENSE text for more details.
//
//    The MIT License(MIT)
//    
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//    
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.
//

namespace KerasSharp.Models
{
    using Accord.Math;
    using KerasSharp.Engine;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Layers;
    using KerasSharp.Losses;
    using KerasSharp.Metrics;
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    using static KerasSharp.Backends.Current;
    using static KerasSharp.Python;
    using KerasSharp.Constraints;
    using KerasSharp.Regularizers;


    /// <summary>
    ///   Linear stack of layers.
    /// </summary>
    /// 
    /// <remarks>
    ///   The first layer passed to a Sequential model should have a defined input shape. What that 
    ///   means is that it should have received an `input_shape` or `batch_input_shape` argument,
    ///   or for some type of layers (recurrent, Dense...) an `input_dim` argument.
    /// </remarks>
    /// 
    /// <example>
    /// <code source="Unit Tests\Accord.Tests.Neuro.TensorFlowSharp\SequentialTest.cs" region="doc_learn" />
    /// </example> 
    /// 
    public class Sequential : Model, IEnumerable<Layer>
    {

        public Model model;


        public Sequential(List<Layer> layers = null, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/models.py#L386

            this.layers = new List<Layer>(); // Stack of layers.
            this.model = null; // Internal Model instance.
            this.inputs = new List<Tensor>(); // List of input tensors
            this.outputs = new List<Tensor>(); // List of length 1: the output tensor (unique).
            this._trainable = true;
            this._initial_weights = null;

            // Model attributes.
            this.inbound_nodes = new List<Node>();
            this.outbound_nodes = new List<Node>();
            this.built = false;

            // Set model name.
            if (name == null)
            {
                string prefix = "sequential_";
                name = prefix + K.get_uid(prefix);
            }

            this.name = name;

            // Add to the model any layers passed to the constructor.
            if (layers != null)
            {
                foreach (Layer layer in layers)
                {
                    this.Add(layer);
                }
            }
        }


        /// <summary>
        /// Adds a layer instance on top of the layer stack.
        /// </summary>
        /// 
        /// <param name="layer">The layer.</param>
        /// 
        public void Add(Layer layer)
        {
            if (outputs.Count == 0)
            {
                // first layer in model: check that it is an input layer
                if (layer.inbound_nodes.Count == 0)
                {
                    // create an input layer
                    if (layer.batch_input_shape == null)
                        throw new Exception("The first layer in a Sequential model must get an 'input_shape' or 'batch_input_shape' argument.");

                    // Instantiate the input layer.
                    var x = Input(batch_shape: layer.batch_input_shape, dtype: layer.dtype, name: $"{layer.name}_input");

                    Debug.Assert(x[0]._keras_history.Value.layer.GetType() == typeof(InputLayer));

                    // This will build the current layer and create the node connecting 
                    // the current layer to the input layer we just created.
                    layer.Call(x);

                    Debug.Assert(x[0]._keras_history.Value.layer.GetType() == typeof(InputLayer));
                }


                if (layer.inbound_nodes.Count != 1)
                {
                    throw new Exception($"A layer added to a Sequential model must not already be connected somewhere else. Model received layer '{layer.name}' which has {layer.inbound_nodes.Count} pre-existing inbound connections.");
                }

                if (layer.inbound_nodes[0].output_tensors.Count != 1)
                {
                    throw new Exception("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");
                }

                this.outputs = new List<Tensor> { layer.inbound_nodes[0].output_tensors[0] };
                this.inputs = base.get_source_inputs(this.outputs[0]);

                // We create an input node, which we will keep updated
                // as we add more layers
                var node = new Node(outbound_layer: this,
                                    inbound_layers: new List<Layer>(),
                                    node_indices: new List<int?>(),
                                    tensor_indices: new List<int?>(),
                                    input_tensors: this.inputs,
                                    output_tensors: this.outputs,
                                    // no model-level masking for now
                                    input_masks: this.inputs.Select(x => (Tensor)null).ToList(),
                                    output_masks: new List<Tensor>() { null },
                                    input_shapes: this.inputs.Select(x => x._keras_shape).ToList(),
                                    output_shapes: this.outputs.Select(x => x._keras_shape).ToList()
                );
            }
            else
            {
                List<Tensor> output_tensor = layer.Call(this.outputs);
                if (output_tensor.Count > 1)
                {
                    throw new Exception("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");
                }

                this.outputs = output_tensor;

                // update this.inbound_nodes
                this.inbound_nodes[0].output_tensors = this.outputs;
                this.inbound_nodes[0].output_shapes = new List<int?[]> { this.outputs[0]._keras_shape };
            }

            this.layers.Add(layer);
            this.built = false;
        }

        /// <summary>
        ///   Removes the last layer in the model.
        /// </summary>
        /// 
        public void pop()
        {
            if (this.layers.Count == 0)
                throw new Exception("There are no layers in the model.");

            this.layers.RemoveAt(this.layers.Count - 1);

            if (this.layers.Count == 0)
            {
                this.outputs = new List<Tensor>();
                this.inbound_nodes = new List<Node>();
                this.outbound_nodes = new List<Node>();
            }
            else
            {
                this.layers[-1].outbound_nodes = new List<Node>();
                this.outputs = new List<Tensor>() { this.layers.ToArray().Get(-1).output[0] };
                // update this.inbound_nodes
            }

            this.inbound_nodes[0].output_tensors = this.outputs;
            this.inbound_nodes[0].output_shapes = new List<int?[]>() { this.outputs[0]._keras_shape };
            this.built = false;
        }

        /// <summary>
        ///   Retrieve a layer that is part of the model. Returns a layer based on 
        ///   either its name(unique) its index in the graph.Indices are based on 
        ///   order of horizontal graph traversal(bottom - up).
        /// </summary>
        /// 
        /// <param name="">The name of layer.</param>
        /// <param name="">The index of layer.</param>
        /// 
        /// <returns> A layer instance.</returns>
        /// 
        public override Layer get_layer(string name = null, int? index = null)
        {
            if (this.model == null)
                this.build();
            return this.model.get_layer(name, index);
        }


        protected override List<Tensor> InnerCall(List<Tensor> inputs, List<Tensor> mask = null, bool? training = null)
        {
            if (this.model == null)
                this.build();
            return this.model.Call(inputs, mask);
        }

        protected override void build(List<int?[]> input_shape = null)
        {
            if (this.inputs == null || this.outputs == null)
                throw new Exception("Sequential model cannot be built: model is empty. Add some layers first.");

            // actually create the model
            this.model = new Model(this.inputs, this.outputs, name: $"{this.name}_model");
            this.model.trainable = this.trainable;

            // mirror model attributes
            this.supports_masking = this.model.supports_masking;
            this._output_mask_cache = this.model._output_mask_cache;
            this._output_tensor_cache = this.model._output_tensor_cache;
            this._output_shape_cache = this.model._output_shape_cache;
            this.input_layers = this.model.input_layers;
            this.input_layers_node_indices = this.model.input_layers_node_indices;
            this.input_layers_tensor_indices = this.model.input_layers_tensor_indices;
            this.output_layers = this.model.output_layers;
            this.output_layers_node_indices = this.model.output_layers_node_indices;
            this.output_layers_tensor_indices = this.model.output_layers_tensor_indices;
            this.nodes_by_depth = this.model.nodes_by_depth;
            this.container_nodes = this.model.container_nodes;
            this.output_names = this.model.output_names;
            this.input_names = this.model.input_names;
            this._feed_input_names = this.model._feed_input_names;
            this._feed_inputs = this.model._feed_inputs;

            // Make sure child model callbacks
            // will call the parent Sequential model.
            this.model.callback_model = this;

            this.built = true;
        }


        public IEnumerator<Layer> GetEnumerator()
        {
            return ((IEnumerable<Layer>)layers).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<Layer>)layers).GetEnumerator();
        }


        public override bool uses_learning_phase
        {
            get
            {
                if (this.model is null)
                    this.build();
                return this.model.uses_learning_phase;
            }
        }

        public override List<Layer> _flattened_layers
        {
            get
            {
                var layers = new List<Layer>();
                if (this.layers != null)
                {
                    // Support for legacy models
                    if (this.layers[0] is Merge)
                    {
                        var merge = this.layers[0] as Merge;
                        foreach (Layer layer in merge.layers)
                        {
                            if (hasattr(layer, "_flattened_layers"))
                            {
                                foreach (Layer sublayer in layer._flattened_layers)
                                {
                                    if (!layers.Contains(sublayer))
                                        layers.Add(sublayer);
                                }
                            }
                            else if (hasattr(layer, "layers"))
                            {
                                foreach (Layer sublayer in merge.layers)
                                {
                                    if (!layers.Contains(sublayer))
                                    {
                                        layers.Add(sublayer);
                                    }
                                    else
                                    {
                                        if (!layers.Contains(layer))
                                            layers.Add(layer);
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if (!layers.Contains(this.layers[0]))
                        {
                            layers.Add(this.layers[0]);
                        }

                        foreach (Layer layer in this.layers.ToArray().Get(1, 0))
                        {
                            if (!layers.Contains(layer))
                                layers.Add(layer);
                        }
                    }
                }

                return layers;
            }
        }

        public List<T> _gather_list_attr<T>(object attr)
        {
            var all_attrs = new List<T>();
            foreach (Layer layer in this._flattened_layers)
                all_attrs.Add(getattr<T>(layer, attr, new object[] { }));
            return all_attrs;
        }



        public override bool trainable
        {
            get { return this._trainable; }
            set
            {
                if (this.model != null)
                    this.model.trainable = value;
                this._trainable = value;
            }
        }




        public override List<Tensor> trainable_weights
        {
            get
            {
                if (!this.trainable)
                    return null;
                // Support for legacy behavior
                return base.trainable_weights;
            }
        }

        public override List<Tensor> non_trainable_weights
        {
            get
            {
                // Support for legacy behavior
                List<Tensor> weights = base.non_trainable_weights;
                if (!this.trainable)
                {
                    trainable_weights = base.trainable_weights;
                    return trainable_weights.Concat(weights).ToList();
                }

                return weights;
            }
        }


        public override List<List<Tensor>> updates
        {
            get
            {
                if (this.model == null)
                    this.build();
                return this.model.updates;
            }
        }

        public override List<List<Tensor>> state_updates()
        {
            if (this.model == null)
                this.build();
            return this.model.state_updates();
        }

        public override List<List<Tensor>> get_updates_for(List<Tensor> inputs)
        {
            if (this.model == null)
                this.build();
            return this.model.get_updates_for(inputs);
        }

        public override List<Tensor> losses
        {
            get
            {
                if (this.model == null)
                    this.build();
                return this.model.losses;
            }
        }

        public override List<Tensor> get_losses_for(List<Tensor> inputs)
        {
            if (this.model == null)
                this.build();
            return this.model.get_losses_for(inputs);
        }

        public override Dictionary<Tensor, IWeightRegularizer> regularizers
        {
            get
            {
                if (this.model == null)
                    this.build();
                return this.model.regularizers;
            }
        }

        public override Dictionary<Tensor, IWeightConstraint> constraints
        {
            get
            {
                if (this.model == null)
                    this.build();
                return this.model.constraints;
            }
        }

        public override List<string> metrics_names
        {
            get
            {
                if (this.model == null)
                    this.build();
                return this.model.metrics_names;
            }
        }

        public override List<Tensor> weights
        {
            get
            {
                //"""Retrieves the weights of the model.
                //# Returns
                //    A flat list of Numpy arrays
                //    (one array per model weight).
                //"""
                // Legacy support
                //if (legacy_models.needs_legacy_support)
                //{
                //    layers = legacy_models.legacy_sequential_layers
                //    weights = []
                //    foreach (Layer layer in layers)
                //        weights.append(layer.get_weights())
                //    return weights;
                //}

                if (this.model == null)
                    this.build();

                return this.model.weights;
            }
        }

        public override void set_weights(List<Array> weights)
        {
            //"""Sets the weights of the model.
            //# Arguments
            //    weights: Should be a list
            //        of Numpy arrays with shapes and types matching
            //        the output of `model.get_weights()`.
            //"""
            // Legacy support
            //if (legacy_models.needs_legacy_support)
            //    layers = legacy_models.legacy_sequential_layers;

            //foreach (Layer layer in layers)
            //{
            //    nb_param = len(layer.weights)
            //    layer.weights = weights.Get(0, nb_param);
            //    weights = weights.Get(nb_param, 0);
            //}

            if (this.model == null)
                this.build();

            this.model.set_weights(weights);
        }


        public override void Compile(IOptimizer optimizer, Dictionary<string, ILoss> loss, Dictionary<string, List<IMetric>> metrics = null,
            Dictionary<string, double> loss_weights = null, Dictionary<string, string> sample_weight_mode = null)
        {
            //"""Configures the learning process.
            //# Arguments
            //    optimizer: str(name of optimizer) or optimizer object.
            //        See[optimizers] (/optimizers).
            //    loss: str(name of objective function) or objective function.
            //       See[losses] (/losses).
            //    metrics: list of metrics to be evaluated by the model
            //        during training and testing.
            //        Typically you will use `metrics=["accuracy"]`.
            //        See[metrics] (/metrics).
            //    sample_weight_mode: if you need to do timestep-wise
            //        sample weighting(2D weights), set this to "temporal".
            //        "null" defaults to sample-wise weights (1D).
            //    **kwargs: for Theano backend, these are passed into K.function.
            //        When using the Tensorflow backend, these are passed into
            //        `tf.Session.run`.
            //# Example
            //    ```python
            //        model = Sequential()
            //        model.add(Dense(32, input_shape=(500,)))
            //        model.add(Dense(10, activation="softmax"))
            //        model.compile(optimizer="rmsprop",
            //                      loss="categorical_crossentropy",
            //                      metrics=["accuracy"])
            //    ```
            //"""
            // create the underlying model
            this.build();
            // call compile method of Model class
            this.model.Compile(optimizer, loss,
                               metrics: metrics,
                               sample_weight_mode: sample_weight_mode);

            this.optimizer = this.model.optimizer;
            this.loss = this.model.loss;
            this.total_loss = this.model.total_loss;
            this.loss_weights = this.model.loss_weights;
            this.metrics = this.model.metrics;
            this.metrics_tensors = this.model.metrics_tensors;
            this.metrics_names = this.model.metrics_names;
            this.sample_weight_mode = this.model.sample_weight_mode;
            this.sample_weights = this.model.sample_weights;
            this.targets = this.model.targets;
        }


        public History fit(Array x = null, Array y = null, int batch_size = 32, int epochs = 1, int verbose = 1,
            CallbackList callbacks = null, double validation_split = 0, IList<Array> validation_data = null, Shuffle shuffle = Shuffle.True,
            Dictionary<int, double> class_weight = null, Array sample_weight = null, int initial_epoch = 0, object kwargs = null, int? nb_epoch = null)
        {
            // Legacy support
            if (nb_epoch != null)
            {
                Trace.TraceWarning("The 'nb_epoch' argument in 'fit' has been renamed 'epochs'.");
                epochs = nb_epoch.Value;
            }

            return this.fit(
                x.dict_from_single(),
                y.dict_from_single(),
                batch_size,
                epochs,
                verbose,
                callbacks,
                validation_split,
                validation_data?.Select(a => a.dict_from_single()).ToList(),
                shuffle,
                class_weight?.Select(p => (p.Key.ToString(), p.Value)).ToDictionary(a => a.Item1, b => b.Item2).dict_from_single(),
                sample_weight.dict_from_single(),
                initial_epoch,
                kwargs);
        }

        public override History fit(Dictionary<string, Array> x = null, Dictionary<string, Array> y = null, int batch_size = 32, int epochs = 1, int verbose = 1,
            CallbackList callbacks = null, double validation_split = 0, IList<Dictionary<string, Array>> validation_data = null, Shuffle shuffle = Shuffle.True,
            Dictionary<string, Dictionary<string, double>> class_weight = null, Dictionary<string, Array> sample_weight = null, int initial_epoch = 0, object kwargs = null)
        {
            //"""Trains the model for a fixed number of epochs.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    y: labels, as a Numpy array.
            //    batch_size: integer.Number of samples per gradient update.
            //   epochs: integer, the number of epochs to train the model.

            //   verbose: 0 for no logging to stdout,
            //        1 for progress bar logging, 2 for one log line per epoch.
            //   callbacks: list of `keras.callbacks.Callback` instances.
            //       List of callbacks to apply during training.
            //       See[callbacks](/callbacks).
            //    validation_split: float (0. < x< 1).
            //        Fraction of the data to use as held-out validation data.
            //    validation_data: tuple (x_val, y_val) or tuple
            //        (x_val, y_val, val_sample_weights) to be used as held-out
            //        validation data.Will override validation_split.
            //    shuffle: boolean or str (for "batch").
            //        Whether to shuffle the samples at each epoch.
            //        "batch" is a special option for dealing with the
            //        limitations of HDF5 data; it shuffles in batch-sized chunks.
            //    class_weight: dictionary mapping classes to a weight value,
            //        used for scaling the loss function (during training only).
            //    sample_weight: Numpy array of weights for
            //        the training samples, used for scaling the loss function
            //        (during training only). You can either pass a flat(1D)
            //        Numpy array with the same length as the input samples
            //        (1:1 mapping between weights and samples),
            //        or in the case of temporal data,
            //        you can pass a 2D array with shape(samples, sequence_length),
            //        to apply a different weight to every timestep of every sample.
            //        In this case you should make sure to specify
            //        sample_weight_mode= "temporal" in compile().
            //    initial_epoch: epoch at which to start training
            //        (useful for resuming a previous training run)
            //# Returns
            //    A `History` object. Its `History.history` attribute is
            //    a record of training loss values and metrics values
            //    at successive epochs, as well as validation loss values
            //    and validation metrics values (if applicable).
            //# Raises
            //    RuntimeError: if the model was never compiled.
            //"""


            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");

            return model.fit(x, y,
                batch_size,
                epochs, verbose,
                callbacks,
                validation_split,
                validation_data,
                shuffle,
                class_weight,
                sample_weight,
                initial_epoch,
                kwargs);
        }

        public override double[] evaluate(Dictionary<string, Array> x, Dictionary<string, Array> y, int batch_size = 32, int verbose = 1, Dictionary<string, Array> sample_weight = null)
        {
            //"""Computes the loss on some input data, batch by batch.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    y: labels, as a Numpy array.
            //    batch_size: integer. Number of samples per gradient update.
            //    verbose: verbosity mode, 0 or 1.
            //    sample_weight: sample weights, as a Numpy array.
            //# Returns
            //    Scalar test loss (if the model has no metrics)
            //    or list of scalars (if the model computes other metrics).
            //    The attribute `model.metrics_names` will give you
            //    the display labels for the scalar outputs.
            //# Raises
            //    RuntimeError: if the model was never compiled.
            //"""
            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");

            return this.model.evaluate(x, y,
                                       batch_size: batch_size,
                                       verbose: verbose,
                                       sample_weight: sample_weight);
        }

        public override Array[] predict(Dictionary<string, Array> x, int batch_size = 32, int verbose = 0)
        {
            //"""Generates output predictions for the input samples.
            //The input samples are processed batch by batch.
            //# Arguments
            //    x: the input data, as a Numpy array.
            //    batch_size: integer.
            //    verbose: verbosity mode, 0 or 1.
            //# Returns
            //    A Numpy array of predictions.
            //"""
            if (this.model == null)
                this.build();

            return this.model.predict(x, batch_size: batch_size, verbose: verbose);
        }

        public override List<Tensor> predict_on_batch(Dictionary<string, Array> x)
        {
            //"""Returns predictions for a single batch of samples.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //# Returns
            //    A Numpy array of predictions.
            //"""
            if (this.model == null)
                this.build();
            return this.model.predict_on_batch(x);
        }

        public override List<Tensor> train_on_batch(Dictionary<string, Array> x, Dictionary<string, Array> y, Dictionary<string, Array> sample_weight = null, Dictionary<string, Dictionary<string, double>> class_weight = null)
        {
            //"""Single gradient update over one batch of samples.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    y: labels, as a Numpy array.
            //    class_weight: dictionary mapping classes to a weight value,
            //        used for scaling the loss function (during training only).
            //    sample_weight: sample weights, as a Numpy array.
            //# Returns
            //    Scalar training loss (if the model has no metrics)
            //    or list of scalars (if the model computes other metrics).
            //    The attribute `model.metrics_names` will give you
            //    the display labels for the scalar outputs.
            //# Raises
            //    RuntimeError: if the model was never compiled.
            //"""
            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");

            return this.model.train_on_batch(x, y,
                                             sample_weight: sample_weight,
                                             class_weight: class_weight);
        }

        public override List<Tensor> test_on_batch(Dictionary<string, Array> x, Dictionary<string, Array> y, Dictionary<string, Array> sample_weight = null)
        {
            //"""Evaluates the model over a single batch of samples.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    y: labels, as a Numpy array.
            //    sample_weight: sample weights, as a Numpy array.
            //# Returns
            //    Scalar test loss (if the model has no metrics)
            //    or list of scalars (if the model computes other metrics).
            //    The attribute `model.metrics_names` will give you
            //    the display labels for the scalar outputs.
            //# Raises
            //    RuntimeError: if the model was never compiled.
            //"""
            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");
            return this.model.test_on_batch(x, y, sample_weight: sample_weight);
        }

        public Array predict_proba(Dictionary<String, Array> x, int batch_size = 32, int verbose = 1)
        {
            //"""Generates class probability predictions for the input samples.
            //The input samples are processed batch by batch.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    batch_size: integer.
            //    verbose: verbosity mode, 0 or 1.
            //# Returns
            //    A Numpy array of probability predictions.
            //"""
            Array preds = this.predict(x, batch_size, verbose);

            //if (preds.Min() < 0.0 || preds.Max() > 1.0)
            //{
            //    Trace.TraceWarning("Network returning invalid probability values. The last layer might not normalize predictions into probabilities (like softmax or sigmoid would).");
            //}

            return preds;
        }

        public Array predict_classes(Dictionary<string, Array> x, int batch_size = 32, int verbose = 1)
        {
            //"""Generate class predictions for the input samples.
            //The input samples are processed batch by batch.
            //# Arguments
            //    x: input data, as a Numpy array or list of Numpy arrays
            //        (if the model has multiple inputs).
            //    batch_size: integer.
            //    verbose: verbosity mode, 0 or 1.
            //# Returns
            //    A numpy array of class predictions.
            //"""
            Array proba = this.predict(x, batch_size: batch_size, verbose: verbose);

            if (proba is double[])
                return (proba as double[]).Apply(i => i > 0.5 ? 1 : 0);

            return (proba as double[][]).ArgMax(dimension: -1);
        }

        public History fit_generator(IEnumerator<List<Dictionary<string, Array>>> generator, int steps_per_epoch,
                          int epochs = 1, int verbose = 1, CallbackList callbacks = null,
                          List<Dictionary<string, Array>> validation_data = null, int? validation_steps = null,
                          Dictionary<int, Tensor> class_weight = null,
                          int max_q_size = 10, int workers = 1, int initial_epoch = 0)
        {
            //        """Fits the model on data generated batch-by-batch by a Python generator.
            //        The generator is run in parallel to the model, for efficiency.
            //        For instance, this allows you to do real-time data augmentation
            //        on images on CPU in parallel to training your model on GPU.
            //# Arguments
            //            generator: A generator.
            //                The output of the generator must be either
            //                - a tuple (inputs, targets)
            //                - a tuple (inputs, targets, sample_weights).
            //                All arrays should contain the same number of samples.
            //                The generator is expected to loop over its data
            //                indefinitely.An epoch finishes when `steps_per_epoch`
            //                batches have been seen by the model.
            //            steps_per_epoch: Total number of steps (batches of samples)
            //                to yield from `generator` before declaring one epoch
            //                finished and starting the next epoch. It should typically
            //                be equal to the number of unique samples of your dataset
            //                divided by the batch size.
            //            epochs: Integer, total number of iterations on the data.
            //            verbose: Verbosity mode, 0, 1, or 2.
            //            callbacks: List of callbacks to be called during training.
            //            validation_data: This can be either
            //                - A generator for the validation data
            //                - A tuple (inputs, targets)
            //                - A tuple (inputs, targets, sample_weights).
            //            validation_steps: Only relevant if `validation_data`
            //                is a generator.
            //                Number of steps to yield from validation generator
            //                at the end of every epoch. It should typically
            //                be equal to the number of unique samples of your
            //                validation dataset divided by the batch size.
            //            class_weight: Dictionary mapping class indices to a weight
            //                for the class.
            //            max_q_size: Maximum size for the generator queue
            //            workers: Maximum number of processes to spin up
            //            pickle_safe: Ff true, use process based threading.
            //                Note that because
            //                this implementation relies on multiprocessing,
            //                you should not pass
            //                non picklable arguments to the generator
            //                as they can"t be passed
            //                easily to children processes.
            //            initial_epoch: Epoch at which to start training
            //                (useful for resuming a previous training run)
            //        # Returns
            //            A `History` object.
            //        # Raises
            //            RuntimeError: if the model was never compiled.
            //        # Example
            //        ```python
            //            public override void generate_arrays_from_file(path):
            //                while 1:
            //                    f = open(path)
            //                    for line in f:
            //                        # create Numpy arrays of input data
            //                        # and labels, from each line in the file
            //                        x, y = process_line(line)
            //                        yield (x, y)
            //                        f.close()
            //            model.fit_generator(generate_arrays_from_file("/my_file.txt"),
            //                                steps_per_epoch=1000, epochs=10)
            //        ```
            //        """
            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");

            return this.model.fit_generator(generator,
                                            steps_per_epoch,
                                            epochs,
                                            verbose: verbose,
                                            callbacks: callbacks,
                                            validation_data: validation_data,
                                            validation_steps: validation_steps,
                                            class_weight: class_weight,
                                            max_queue_size: max_q_size,
                                            workers: workers,
                                            //pickle_safe: pickle_safe,
                                            initial_epoch: initial_epoch);
        }

        public List<Tensor> evaluate_generator(IEnumerator<List<List<Tensor>>> generator, int steps,
                               int max_q_size = 10, int workers = 1,
                               bool pickle_safe = false)
        {
            //"""Evaluates the model on a data generator.
            //The generator should return the same kind of data
            //as accepted by `test_on_batch`.
            //# Arguments
            //    generator: Generator yielding tuples (inputs, targets)
            //        or (inputs, targets, sample_weights)
            //    steps: Total number of steps (batches of samples)
            //        to yield from `generator` before stopping.
            //    max_q_size: maximum size for the generator queue
            //    workers: maximum number of processes to spin up
            //    pickle_safe: if true, use process based threading.
            //        Note that because this implementation
            //        relies on multiprocessing, you should not pass
            //        non picklable arguments to the generator
            //        as they can"t be passed easily to children processes.
            //# Returns
            //    Scalar test loss (if the model has no metrics)
            //    or list of scalars (if the model computes other metrics).
            //    The attribute `model.metrics_names` will give you
            //    the display labels for the scalar outputs.
            //# Raises
            //    RuntimeError: if the model was never compiled.
            //"""
            if (this.model == null)
                throw new InvalidOperationException("The model needs to be compiled before being used.");

            return this.model.evaluate_generator(generator,
                                             steps,
                                             max_queue_size: max_q_size,
                                             workers: workers
                                             //pickle_safe: pickle_safe
                                             );
        }

        public List<List<Tensor>> predict_generator(IEnumerator<List<Dictionary<string, Array>>> generator,
            int steps, int max_q_size = 10, int workers = 1, int verbose = 0)
        {
            //"""Generates predictions for the input samples from a data generator.
            //The generator should return the same kind of data as accepted by
            //`predict_on_batch`.
            //# Arguments
            //    generator: generator yielding batches of input samples.
            //    steps: Total number of steps (batches of samples)
            //        to yield from `generator` before stopping.
            //    max_q_size: maximum size for the generator queue
            //    workers: maximum number of processes to spin up
            //    pickle_safe: if true, use process based threading.
            //        Note that because this implementation
            //        relies on multiprocessing, you should not pass
            //        non picklable arguments to the generator
            //        as they can"t be passed easily to children processes.
            //    verbose: verbosity mode, 0 or 1.
            //# Returns
            //    A Numpy array of predictions.
            //"""
            if (this.model == null)
                this.build();

            return this.model.predict_generator(generator, steps,
                                                max_queue_size: max_q_size,
                                                workers: workers,
                                                verbose: verbose);
        }

    }
}