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

namespace KerasSharp.Engine.Topology
{
    using System;
    using Accord.Math;
    using KerasSharp.Constraints;
    using KerasSharp.Initializers;
    using KerasSharp.Regularizers;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    

    using static KerasSharp.Backends.Current;
    using static KerasSharp.Python;



    /// <summary>
    /// Abstract base layer class.
    /// </summary>
    /// 
    public abstract class Layer
    {
        public virtual List<InputSpec> input_spec { get; set; }
        public List<Tensor> _trainable_weights;
        public List<Tensor> _non_trainable_weights;
        public bool _trainable;
        public List<Array> _initial_weights;
        public Dictionary<Tensor, IWeightConstraint> _constraints;
        public List<Tensor> _losses;
        public List<List<Tensor>> _updates;
        public Dictionary<string, List<Tensor>> _per_input_losses;
        public Dictionary<string, List<List<Tensor>>> _per_input_updates;
        public bool _built;

        public bool supports_masking;

        public virtual bool trainable
        {
            get { return _trainable; }
            set { _trainable = value; }
        }

        public string name;
        public virtual bool uses_learning_phase { get; set; }
        protected internal IWeightRegularizer activity_regularizer;

        public List<Node> outbound_nodes;
        public List<Node> inbound_nodes;
        protected internal int?[] batch_input_shape;
        protected internal DataType? input_dtype;
        public bool is_placeholder;


        protected internal DataType dtype;

        public virtual List<Layer> _flattened_layers { get; set; }

        public virtual bool stateful { get; set; }




        /// <summary>
        ///   Creates a new Layer.
        /// </summary>
        /// 
        /// <param name="name">The name of this layer. Must be unique within a model.</param>
        /// <param name="input_spec">A list of InputSpec class instances where each entry describes 
        ///   one required input: 1) ndim 2) dtype. A layer with `n` input tensors must have an 
        ///   `input_spec` of length `n`.</param>
        /// <param name="trainable">Whether the layer weights will be updated during training.</param>
        /// <param name="uses_learning_phase">Whether any operation of the layer uses `K.in_training_phase()` or `K.in_test_phase()`.</param>
        /// <param name="input_shape">Provided for convenience, but note that there may be cases in
        ///   which this attribute is ill-defined (e.g.a shared layer with multiple input shapes), in which
        ///   case requesting `input_shape` will raise an Exception. Prefer using `layer.get_input_shape_for(input_shape)`,
        ///   or `layer.get_input_shape_at(node_index)`.</param>
        /// <param name="output_shape"></param>
        /// <param name="inbound_nodes">List of nodes.</param>
        /// <param name="outbound_nodes">List of nodes.</param>
        /// <param name="input">Input tensor(s). Note that if the layer is used more than once (shared layer), this is ill-defined and will raise an exception. In such cases, use 'layer.get_input_at(node_index)'.</param>
        /// <param name="output">Output tensor(s). Note that if the layer is used more than once (shared layer), this is ill-defined and will raise an exception. In such cases, use 'layer.get_input_at(node_index)'.</param>
        /// <param name="input_mask">Same as <paramref="input"/>, but for masks.</param>
        /// <param name="output_mask">Same as <paramref="output"/>, but for masks.</param>
        /// <param name="trainable_weights">List of variables.</param>
        /// <param name="non_trainable_weights">List of variables.</param>
        /// <param name="weights">The concatenation of the lists trainable_weights and non_trainable_weights (in this order).</param>
        /// <param name="dtype"></param>
        /// <param name="constraints">Dictionary mapping weights to constraints.</param>
        /// <param name="batch_input_shape"></param>
        /// <param name="batch_size"></param>
        /// 
        public Layer(string name = null, List<InputSpec> input_spec = null, bool trainable = true,
            bool uses_learning_phase = true, int?[] input_shape = null, int[] output_shape = null,
            List<Node> inbound_nodes = null, List<Node> outbound_nodes = null, Tensor input = null,
            Tensor output = null, Tensor input_mask = null, Tensor output_mask = null, List<Tensor> trainable_weights = null,
            List<Array> non_trainable_weights = null, Tensor weights = null, DataType? dtype = null,
            Dictionary<Tensor, IWeightConstraint> constraints = null, int?[] batch_input_shape = null,
            int? batch_size = null, int? input_dim = null)
        {
            if (input_shape == null && input_dim != null)
                input_shape = new int?[] { input_dim };

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L247

            this.input_spec = null;
            this.supports_masking = false;

            // These properties will be set upon call of this.build()
            this._trainable_weights = new List<Tensor>();
            this._non_trainable_weights = new List<Tensor>();
            this._constraints = new Dictionary<Tensor, IWeightConstraint>();  // dict {tensor: constraint instance}
            this._losses = new List<Tensor>();
            this._updates = new List<List<Tensor>>();
            this._per_input_losses = new Dictionary<string, List<Tensor>>();
            this._per_input_updates = new Dictionary<string, List<List<Tensor>>>();
            this._built = false;

            // These lists will be filled via successive calls
            // to this._add_inbound_node().
            this.inbound_nodes = new List<Node>();
            this.outbound_nodes = new List<Node>();

            // These properties should be set by the user via keyword arguments.
            // note that "dtype", "input_shape" and "batch_input_shape"
            // are only applicable to input layers: do not pass these keywords
            // to non-input layers.
            //allowed_kwargs = {
            //    "input_shape",
            //              "batch_input_shape",
            //              "batch_size",
            //              "dtype",
            //              "name",
            //              "trainable",
            //              "weights",
            //              "input_dtype",  // legacy
            //              }

            if (name == null)
            {
                string prefix = this.GetType().Name;
                name = _to_snake_case(prefix) + "_" + K.get_uid(prefix);
            }

            this.name = name;

            this._trainable = trainable;

            if (input_shape != null || batch_input_shape != null)
            {
                // In this case we will later create an input layer
                // to insert before the current layer
                if (input_shape != null)
                    batch_input_shape = new[] { batch_size }.Concatenate(input_shape);
                this.batch_input_shape = batch_input_shape;
            }

            // Set dtype.
            if (dtype == null)
                dtype = input_dtype;
            if (dtype == null)
                dtype = K.floatx();
            this.dtype = dtype.Value;

            if (weights != null)
                this._initial_weights = non_trainable_weights;
        }

        private static string _to_snake_case(string s)
        {
            var re = "(.)([A-Z][a-z0-9]+)";
            var intermediate = System.Text.RegularExpressions.Regex.Replace(s, re, "$1_$2");

            var insecurePattern = "([a-z])([A-Z])";
            var insecure = System.Text.RegularExpressions.Regex.Replace(intermediate, insecurePattern, "$1_$2").ToLower();
            /*
			 In Python, a class starting with "_" is insecure for creating scopes. 
			 While this is not a concern in C#, it's implemented here for compatibility with Keras naming.
			*/
            if (insecure.StartsWith("_", StringComparison.InvariantCulture))
            {
                return "private" + insecure;
            }
            else
            {
                return insecure;
            }
        }

        public virtual List<Tensor> losses
        {
            get { return this._losses; }
        }

        public virtual List<List<Tensor>> updates
        {
            get { return this._updates; }
        }

        public bool built
        {
            get { return this._built; }
            set { this._built = value; }
        }

        public virtual Dictionary<Tensor, IWeightConstraint> constraints
        {
            get { return this._constraints; }
            set { this._constraints = value; }
        }

        public virtual List<Tensor> trainable_weights
        {
            get
            {
                if (trainable)
                    return this._trainable_weights;
                return new List<Tensor> { };
            }
            set
            {
                this._trainable_weights = value;
            }
        }

        public virtual List<Tensor> non_trainable_weights
        {
            get
            {
                if (!trainable)
                    return this._trainable_weights.Concat(this._non_trainable_weights).ToList();
                return this._non_trainable_weights;
            }
            set
            {
                this._non_trainable_weights = value;
            }
        }

        /// <summary>
        ///   Adds a weight variable to the layer.
        /// </summary>
        /// <param name="name">The name for the weight variable.</param>
        /// <param name="shape">The shape tuple of the weight.</param>
        /// <param name="dtype">The <see cref="TFDataType"/> of the weight.</param>
        /// <param name="initializer">An Initializer instance.</param>
        /// <param name="regularizer">An optional Regularizer instance.</param>
        /// <param name="trainable">A boolean, whether the weight should be trained via backprop 
        ///   or not (assuming that the layer itself is also trainable).</param>
        /// <param name="constraint">An optional Constraint instance.</param>
        /// 
        /// <return>The created weight variable.</return>
        /// 
        public Tensor add_weight(string name, int?[] shape, DataType? dtype = null,
            IWeightInitializer initializer = null, IWeightRegularizer regularizer = null,
                   bool trainable = true, IWeightConstraint constraint = null)
        {
            Tensor weight = K.variable(tensor: initializer.Call(shape), dtype: dtype, name: name);

            if (regularizer != null)
                this.add_loss(new List<Tensor>() { regularizer.Call(weight) });

            if (constraint != null)
                this.constraints[weight] = constraint;

            if (trainable)
                this._trainable_weights.Add(weight);
            else
                this._non_trainable_weights.Add(weight);

            return weight;
        }


        /// <summary>
        ///   Checks compatibility between the layer and provided inputs.
        /// </summary>
        /// 
        /// <remarks>
        ///   This checks that the tensor(s) `input` verify the input assumptions of the layer (if any). If not, exceptions are raised.
        /// </remarks>
        /// 
        /// <param name="inputs">The input tensor or list of input tensors.</param>
        /// 
        public void assert_input_compatibility(List<Tensor> inputs)
        {
            // https://github.com/fchollet/keras/blob/2382f788b4f14646fa8b6b2d8d65f1fc138b35c4/keras/engine/topology.py#L393

            if (this.input_spec == null)
                return;

            if (inputs.Count != input_spec.Count)
                throw new Exception("Layer {this.name} expects " + input_spec.Count + " inputs, but it received " + inputs.Count + " input tensors. Input received: " + inputs);

            for (int input_index = 0; input_index < inputs.Count; input_index++)
            {
                Tensor x = inputs[input_index];
                InputSpec spec = input_spec[input_index];

                if (spec == null)
                    continue;

                // Check ndim.
                if (spec.ndim != null)
                    if (K.ndim(x) != spec.ndim)
                        throw new Exception($"Input {input_index} is incompatible with layer {this.name}: expected ndim={spec.ndim}, found ndim={K.ndim(x)}");

                int? ndim = null;
                if (spec.max_ndim != null)
                    ndim = K.ndim(x);

                if (ndim != null && ndim > spec.max_ndim)
                    throw new Exception($"Input {input_index} is incompatible with layer {this.name}: expected max_ndim={ spec.max_ndim }, found ndim={K.ndim(x)}");

                if (spec.min_ndim != null)
                {
                    ndim = K.ndim(x);
                    if (ndim != null && ndim < spec.min_ndim)
                        throw new Exception($"Input {input_index} is incompatible with layer {this.name}: expected min_ndim={spec.min_ndim}, found ndim={K.ndim(x)}");
                }

                // Check dtype.
                if (spec.dtype != null)
                {
                    if (K.dtype(x) != spec.dtype)
                        throw new Exception($"Input {input_index} is incompatible with layer {this.name}: expected dtype={spec.dtype}, found dtype={K.dtype(x)}");
                }

                // Check specific shape axes.
                // https://github.com/fchollet/keras/blob/2382f788b4f14646fa8b6b2d8d65f1fc138b35c4/keras/engine/topology.py#L467
                if (spec.axes != null)
                {
                    int?[] x_shape = K.int_shape(x);

                    if (x_shape != null)
                    {
                        foreach (KeyValuePair<int, int> pair in spec.axes)
                        {
                            int axis = pair.Key;
                            int? value = pair.Value;

                            if (value != null)
                            {
                                int? v = x_shape.Get(axis);
                                if (v != value && v != null)
                                {
                                    throw new Exception($"Input {input_index} is incompatible with layer {this.name}: expected " +
                                        $"axis {axis} of input shape to have value {value} but got shape {x_shape}.");
                                }
                            }
                        }
                    }
                }

                // Check shape.
                if (spec.shape != null)
                {
                    int?[] x_shape = K.int_shape(x);

                    if (x_shape != null)
                    {
                        for (int i = 0; i < spec.shape.Length; i++)
                        {
                            int? spec_dim = spec.shape[i];
                            int? dim = x_shape[i];
                            if (spec_dim != null && dim != null)
                            {
                                if (spec_dim != dim)
                                    throw new Exception($"Input {input_index} is incompatible with layer {this.name} expected shape={spec.shape}, found shape={x_shape}");
                            }
                        }
                    }
                }
            }
        }

        public List<Tensor> this[params Tensor[] x]
        {
            get { return this.Call(x.ToList()); }
        }

        ///  <summary>
        ///    Wrapper around this.call(), for handling internal references.
        ///  </summary>
        ///  
        ///  <remarks>
        ///  If a Keras tensor is passed:
        ///  -We call this._add_inbound_node().
        ///  -If necessary, we `build` the layer to match the _keras_shape of the input(s).
        ///  -We update the _keras_shape of every input tensor with its new shape(obtained via this.compute_output_shape). This is done as part of _add_inbound_node().
        ///  -We update the _keras_history of the output tensor(s) with the current layer. This is done as part of _add_inbound_node().
        ///  </remarks>
        ///  
        ///  <param name="inputs">Can be a tensor or list/ tuple of tensors.</param>
        ///  
        ///  <returns>Output of the layer"s `call` method.</returns>.
        ///  
        public List<Tensor> Call(Tensor input, Tensor mask = null, bool? training = null)
        {
            if (mask == null)
                return Call(new List<Tensor> { input }, null, training);
            return Call(new List<Tensor> { input }, new List<Tensor> { mask }, training);
        }

        ///  <summary>
        ///    Wrapper around this.call(), for handling internal references.
        ///  </summary>
        ///  
        ///  <remarks>
        ///  If a Keras tensor is passed:
        ///  -We call this._add_inbound_node().
        ///  -If necessary, we `build` the layer to match the _keras_shape of the input(s).
        ///  -We update the _keras_shape of every input tensor with its new shape(obtained via this.compute_output_shape). This is done as part of _add_inbound_node().
        ///  -We update the _keras_history of the output tensor(s) with the current layer. This is done as part of _add_inbound_node().
        ///  </remarks>
        ///  
        ///  <param name="inputs">Can be a tensor or list/ tuple of tensors.</param>
        ///  
        ///  <returns>Output of the layer"s `call` method.</returns>.
        ///  
        public List<Tensor> Call(List<Tensor> inputs, List<Tensor> mask = null, bool? training = null)
        {
            using (K.name_scope(this.name))
            {
                // Handle laying building (weight creating, input spec locking).
                if (!this.built)
                {
                    // Raise exceptions in case the input != compatible
                    // with the input_spec specified in the layer constructor.
                    this.assert_input_compatibility(inputs);
                }

                // Collect input shapes to build layer.
                var input_shapes = new List<int?[]>();
                foreach (Tensor x_elem in inputs)
                {
                    if (x_elem._keras_shape != null)
                        input_shapes.Add(x_elem._keras_shape);
                    else if (x_elem.int_shape != null)
                        input_shapes.Add(K.int_shape(x_elem));
                    else
                        throw new Exception($"You tried to call layer {this.name}. This layer has no information about its expected input shape, and thus cannot be built. You can build it manually via: `layer.build(batch_input_shape)`");
                }

                this.build(input_shapes);
                this.built = true;

                // Load weights that were specified at layer instantiation.
                if (this._initial_weights != null)
                    this.set_weights(this._initial_weights);

                // Raise exceptions in case the input != compatible
                // with the input_spec set at build time.
                this.assert_input_compatibility(inputs);

                // Handle mask propagation.
                List<Tensor> previous_mask = Container._collect_previous_mask(inputs);
                //var user_kwargs = copy.copy(kwargs);

                List<Tensor> nextMask = null;
                if (previous_mask.Any((Tensor x) => x != null))
                {
                    // The previous layer generated a mask.
                    if (mask != null)
                    {
                        // If mask is explicitly passed to __call__,
                        // we should override the default mask.
                        nextMask = previous_mask;
                    }
                }

                // Handle automatic shape inference (only useful for Theano).
                List<int?[]> input_shape = Container._collect_input_shape(inputs);

                // Actually call the layer, collecting output(s), mask(s), and shape(s).
                List<Tensor> output = this.InnerCall(inputs, nextMask, training);
                List<Tensor> output_mask = this.compute_mask(inputs, previous_mask);

                // If the layer returns tensors from its inputs, unmodified,
                // we copy them to avoid loss of tensor metadata.
                var output_ls = new List<Tensor>(output);
                var inputs_ls = new List<Tensor>(inputs);
                var output_ls_copy = new List<Tensor>();
                foreach (Tensor x in output_ls)
                {
                    if (inputs_ls.Contains(x))
                        output_ls_copy.Add(K.identity(x));
                    else output_ls_copy.Add(x);
                }

                List<int?[]> output_shape;

                // Infering the output shape is only relevant for Theano.
                if (input_shape.All(s => s != null))
                {
                    output_shape = this.compute_output_shape(input_shape);
                }
                else
                {
                    output_shape = input_shape.Select(x => (int?[])null).ToList();
                }

                //if (output_ls.Count > 1)
                //{
                //    // Augment the mask to match the length of the output.
                //    output_mask = output_ls.Select(x => output_mask).ToList();
                //}

                // Add an inbound node to the layer, so that it keeps track
                // of the call and of all new variables created during the call.
                // This also updates the layer history of the output tensor(s).
                // If the input tensor(s) had not previous Keras history,
                // this does nothing.
                this._add_inbound_node(input_tensors: inputs, output_tensors: output,
                                        input_masks: previous_mask, output_masks: output_mask,
                                        input_shapes: input_shape, output_shapes: output_shape
                                        //arguments: user_kwargs
                                        );

                // Apply activity regularizer if any:
                if (this.activity_regularizer != null)
                {
                    List<Tensor> regularization_losses = this.activity_regularizer.Call(output);
                    this.add_loss(regularization_losses, inputs);
                }

                return output;
            }
        }


        /// <summary>
        ///   This is where the layer's logic lives.
        /// </summary> 
        /// 
        /// <param name="inputs">Input tensor, or list of input tensors.</param>
        /// 
        /// <returns>A tensor or list of tensors.</returns>
        /// 
        protected virtual List<Tensor> InnerCall(List<Tensor> inputs, List<Tensor> mask = null, bool? training = null)
        {
            if (inputs.Count != 1)
                throw new InvalidCastException();

            return new List<Tensor>() { InnerCall(inputs[0], mask?[0], training) };
        }

        /// <summary>
        ///   This is where the layer's logic lives.
        /// </summary> 
        /// 
        /// <param name="inputs">Input tensor, or list of input tensors.</param>
        /// 
        /// <returns>A tensor or list of tensors.</returns>
        /// 
        protected virtual Tensor InnerCall(Tensor inputs, Tensor mask = null, bool? training = null)
        {
            return inputs;
        }

        /// <summary>
        ///   Internal method to create an inbound node for the layer.
        /// </summary>
        /// 
        /// <param name="input_tensors">List of input tensors.</param>
        /// <param name="output_tensors">List of output tensors.</param>
        /// <param name="input_masks">List of input masks (a mask can be a tensor, or null).</param>
        /// <param name="output_masks">List of output masks (a mask can be a tensor, or null).</param>
        /// <param name="input_shapes">List of input shape arrays.</param>
        /// <param name="input_shapes">List of output shape arrays.</param>
        /// <param name="arguments">Dictionary of keyword arguments that were passed to the `call` method of the layer at the call that created the node.</param>
        /// 
        private void _add_inbound_node(List<Tensor> input_tensors, List<Tensor> output_tensors, List<Tensor> input_masks,
            List<Tensor> output_masks, List<int?[]> input_shapes, List<int?[]> output_shapes, object arguments = null)
        {
            // Collect input tensor(s) coordinates.
            var inbound_layers = new List<Layer>();
            var node_indices = new List<int?>();
            var tensor_indices = new List<int?>();
            foreach (Tensor x in input_tensors)
            {
                if (x._keras_history != null)
                {
                    var (inbound_layer, node_index, tensor_index) = x._keras_history.Value;
                    inbound_layers.Add(inbound_layer);
                    node_indices.Add(node_index);
                    tensor_indices.Add(tensor_index);
                }
                else
                {
                    inbound_layers.Add(null);
                    node_indices.Add(null);
                    tensor_indices.Add(null);
                }
            }

            // Create node, add it to inbound nodes.
            new Node(this, inbound_layers: inbound_layers,
                node_indices: node_indices,
                tensor_indices: tensor_indices,
                input_tensors: input_tensors,
                output_tensors: output_tensors,
                input_masks: input_masks,
                output_masks: output_masks,
                input_shapes: input_shapes,
                output_shapes: output_shapes,
                arguments: arguments);

            // Update tensor history, _keras_shape and _uses_learning_phase.
            for (int i = 0; i < output_tensors.Count; i++)
            {
                output_tensors[i]._keras_shape = output_shapes[i];
                bool uses_lp = input_tensors.Any(x => x._uses_learning_phase);
                uses_lp = this.uses_learning_phase || uses_lp;
                output_tensors[i]._uses_learning_phase = output_tensors[i]._uses_learning_phase || uses_lp;
                output_tensors[i]._keras_history = (this, this.inbound_nodes.Count - 1, i);
            }
        }

        /// <summary>
        ///   Computes the output shape of the layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Assumes that the layer will be built to match that input shape provided.
        /// </remarks>
        /// 
        /// <param name="input_shape">Shape array  or list of shape tuples(one per output tensor of the layer). Shape tuples can include null for free dimensions, instead of an integer.</param>
        /// 
        /// <returns>An input shape tuple.</returns>
        /// 
        public List<int?[]> compute_output_shape(int?[] input_shape)
        {
            return compute_output_shape(new List<int?[]>() { input_shape });
        }

        /// <summary>
        ///   Computes the output shape of the layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Assumes that the layer will be built to match that input shape provided.
        /// </remarks>
        /// 
        /// <param name="input_shape">Shape array  or list of shape tuples(one per output tensor of the layer). Shape tuples can include null for free dimensions, instead of an integer.</param>
        /// 
        /// <returns>An input shape tuple.</returns>
        /// 
        public virtual List<int?[]> compute_output_shape(List<int?[]> input_shape)
        {
            //Trace.Warning("Class `{}.{}` defines `get_output_shape_for` but does not override `compute_output_shape`. If this is a Keras 1 layer, please implement `compute_output_shape` to support Keras 2.");
            return input_shape;
        }

        public virtual void reset_states()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Computes an output mask tensor.
        /// </summary>
        /// 
        /// <param name="inputs">Tensor or list of tensors.</param>
        /// <param name="mask">Tensor or list of tensors.</param>
        /// 
        /// <returns>null or a tensor (or list of tensors, one per output tensor of the layer).</returns>
        /// 
        public virtual List<Tensor> compute_mask(Tensor inputs, Tensor mask = null)
        {
            if (mask == null)
                return compute_mask(new List<Tensor>() { inputs });
            return compute_mask(new List<Tensor>() { inputs }, new List<Tensor>() { mask });
        }

        /// <summary>
        ///   Computes an output mask tensor.
        /// </summary>
        /// 
        /// <param name="inputs">Tensor or list of tensors.</param>
        /// <param name="mask">Tensor or list of tensors.</param>
        /// 
        /// <returns>null or a tensor (or list of tensors, one per output tensor of the layer).</returns>
        /// 
        public virtual List<Tensor> compute_mask(List<Tensor> inputs, List<Tensor> mask = null)
        {
            if (!this.supports_masking)
            {
                if (mask != null && !(mask.Count == 1 && mask[0] == null))
                {
                    foreach (Tensor m in mask)
                        throw new Exception("Layer {this.name} does not support masking, but was passed an input_mask: " + m);
                }
            }

            //if masking is explictly supported, by default
            // carry over the input mask
            return mask;
        }

        /// <summary>
        ///   Creates the layer weights.
        /// </summary>
        /// 
        /// <remarks>
        ///   Must be implemented on all layers that have weights.
        /// </remarks>
        /// 
        /// <param name="input_shape">Keras tensor (future input to layer) or list/ tuple of Keras tensors to reference for weight shape computations.</param>
        /// 
        protected virtual void build(List<int?[]> input_shape)
        {
            this.built = true;
        }

        ///// <summary>
        ///// Retrieves an attribute (e.g. input_tensors) from a node.
        ///// </summary>
        ///// <remarks>
        /////    This is used to implement the methods: get_input_shape_at, get_output_shape_at, get_input_at, etc.
        ///// </remarks>
        ///// <param name="node_index">Index of the node.</param>
        ///// <param name="attr">The exact node attribute name.</param>
        ///// <param name="attr_name">Human - readable attribute name, for error messages.</param>
        ///// <returns>The layer's attribute `attr` at the node of index `node_index`.</returns>
        //public object _get_node_attribute_at_index(int node_index, string attr, string attr_name)
        //{
        //    if (this.inbound_nodes.Count == 0)
        //        throw new Exception("The layer has never been called and thus has no defined {attr_name}.");
        //    if (this.inbound_nodes.Count > node_index)
        //        throw new Exception($"Asked to get {attr_name} at node {node_index}, but the layer has only {this.inbound_nodes.Count} inbound nodes.");
        //    var values = getattr(this.inbound_nodes[node_index], attr);
        //    return values;
        //}

        /// <summary>
        ///   Retrieves the input shape(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute.
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        /// 
        /// <returns>A shape array (or list of arrays if the layer has multiple inputs).</returns>
        /// 
        public object get_input_shape_at(int node_index)
        {
            return this.inbound_nodes[node_index].input_shapes;
        }

        /// <summary>
        ///   Retrieves the output shape(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute. 
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        ///   
        /// <returns>A shape array (or list of arrays if the layer has multiple inputs).</returns>
        /// 
        public List<int?[]> get_output_shape_at(int node_index)
        {
            return this.inbound_nodes[node_index].output_shapes;
        }

        /// <summary>
        ///   Retrieves the input tensor(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute. 
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        ///   
        /// <returns>A tensor(or list of tensors if the layer has multiple inputs).</returns>
        /// 
        public List<Tensor> get_input_at(int node_index)
        {
            return this.inbound_nodes[node_index].input_tensors;
        }

        /// <summary>
        ///   Retrieves the output tensor(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute. 
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        ///   
        /// <returns>A tensor(or list of tensors if the layer has multiple outputs).</returns>
        /// 
        public List<Tensor> get_output_at(int node_index)
        {
            return this.inbound_nodes[node_index].output_tensors;
        }

        /// <summary>
        /// Retrieves the input mask tensor(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute. 
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        ///   
        /// <returns>A mask tensor (or list of tensors if the layer has multiple inputs).</returns>
        /// 
        public List<Tensor> get_input_mask_at(int node_index)
        {
            return this.inbound_nodes[node_index].input_masks;
        }
        /// <summary>
        /// Retrieves the output mask tensor(s) of a layer at a given node.
        /// </summary>
        /// 
        /// <param name="node_index">The index of the node from which to retrieve the attribute. 
        ///   E.g. `node_index = 0` will correspond to the first time the layer was called.</param>
        ///   
        /// <returns>A mask tensor (or list of tensors if the layer has multiple outputs).</returns>
        /// 
        public List<Tensor> get_output_mask_at(int node_index)
        {
            return this.inbound_nodes[node_index].output_masks;
        }

        /// <summary>
        ///   Retrieves the input tensor(s) of a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer.
        /// </remarks>
        /// 
        /// <returns>Input tensor or list of input tensors.</returns>
        /// 
        public List<Tensor> input
        {
            get
            {
                if (this.inbound_nodes.Count > 1)
                    throw new Exception($"Layer {this.name} has multiple inbound nodes, hence the notion of 'layer input' is ill-defined. Use `get_input_at(node_index)` instead.");
                else if (this.inbound_nodes.Count == 0)
                    throw new Exception($"Layer {this.name} != connected, no input to return.");
                return this.inbound_nodes[0].input_tensors;
            }
        }

        /// <summary>
        ///   Retrieves the output tensor(s) of a layer.
        /// </summary>
        /// 
        /// <remarks> Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer.</remarks>
        /// 
        /// <returns>Output tensor or list of output tensors.</returns>
        /// 
        public List<Tensor> output
        {
            get
            {
                if (this.inbound_nodes.Count == 0)
                    throw new Exception($"Layer {this.name} has no inbound nodes.");
                if (this.inbound_nodes.Count > 1)
                    throw new Exception($"Layer {this.name} has multiple inbound nodes, hence the notion of \"layer output\" is ill-defined. Use `get_output_at(node_index)` instead.");
                return this.inbound_nodes[0].output_tensors;
            }
        }


        /// <summary>
        ///   Retrieves the input mask tensor(s) of a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer.
        /// </remarks>
        /// 
        /// <returns>Input mask tensor (potentially null) or list of input mask tensors.</returns>
        /// 
        public List<Tensor> input_mask
        {
            get
            {
                if (this.inbound_nodes.Count != 1)
                    throw new Exception($"Layer {this.name} has multiple inbound nodes, hence the notion of \"layer input mask\" is ill-defined. Use `get_input_mask_at(node_index)` instead.");
                return this.inbound_nodes[0].input_mask;
            }
        }

        /// <summary>
        ///   Retrieves the output mask tensor(s) of a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer.
        /// </remarks>
        /// 
        /// <returns>Output mask tensor (potentially null) or list of output mask tensors.</returns>
        /// 
        public List<Tensor> output_mask
        {
            get
            {
                if (this.inbound_nodes.Count != 1)
                    throw new Exception($"Layer {this.name} has multiple inbound nodes, hence the notion of \"layer output mask\" is ill-defined. Use `get_output_mask_at(node_index)` instead.");
                return this.inbound_nodes[0].output_mask;
            }
        }

        /// <summary>
        ///   Retrieves the input shape tuple(s) of a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Only applicable if the layer has exactly one inbound node, i.e. if it is connected to one incoming layer.
        /// </remarks>
        /// 
        /// <returns>Input shape tuple (or list of input shape tuples, one tuple per input tensor).</returns>
        /// 
        public List<int?[]> input_shape
        {
            get
            {
                if (this.inbound_nodes.Count == 0)
                    throw new Exception("The layer has never been called and thus has no defined input shape.");

                var all_input_shapes = new HashSet<string>(this.inbound_nodes.Select(n => str(n.input_shapes)));
                if (all_input_shapes.Count == 1)
                    return this.inbound_nodes[0].input_shapes;

                throw new Exception("The layer {this.name} has multiple inbound nodes, with different input shapes. Hence the notion of \"input shape\" is ill-defined for the layer. Use `get_input_shape_at(node_index)` instead.");
            }
        }





        /// <summary>
        ///   Retrieves the output shape tuple(s) of a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Only applicable if the layer has one inbound node, or if all inbound nodes have the same output shape.
        /// </remarks>
        /// 
        /// <returns>Output shape tuple (or list of input shape tuples, one tuple per output tensor).</returns>
        /// 
        public List<int?[]> output_shape
        {
            get
            {
                if (this.inbound_nodes.Count == 0)
                    throw new Exception("The layer has never been called and thus has no defined output shape.");

                var all_output_shapes = new HashSet<string>(this.inbound_nodes.Select(n => str(n.output_shapes)));
                if (all_output_shapes.Count == 1)
                    return this.inbound_nodes[0].output_shapes;

                throw new Exception("The layer {this.name} has multiple inbound nodes, with different output shapes. Hence the notion of \"input shape\" is ill-defined for the layer. Use `get_output_shape_at(node_index)` instead.");
            }
        }

        /// <summary>
        ///   Add losses to the layer.
        /// </summary>
        /// 
        /// <param name="losses">The loss tensor or list of loss tensors to add to the layer.</param>
        /// <param name="inputs">The input tensor or list of inputs tensors to mark the losses as
        ///   conditional on these inputs. If null is passed, the loss is assumed unconditional (e.g.L2 
        ///   weight regularization, which only depends on the layer"s weights variables, not on any inputs tensors).</param>
        /// 
        /// <remarks>The loss may potentially be conditional on some inputs tensors, for instance activity losses are conditional on the layer"s inputs.</remarks>
        /// 
        public void add_loss(List<Tensor> losses, List<Tensor> inputs = null)
        {
            if (losses == null || losses.Count == 0)
                return;

            // Update this.losses
            if (this._losses != null)
            {
                this._losses.AddRange(losses);

                // Update this._per_input_updates
                if (inputs != null && inputs.Count == 0)
                    inputs = null;
                string inputs_hash;
                if (inputs != null)
                {
                    inputs_hash = _object_list_uid(inputs);
                }
                else
                {
                    // Updates indexed by null are unconditional
                    // rather than input-dependent
                    inputs_hash = null;
                }

                if (!this._per_input_losses.ContainsKey(inputs_hash))
                    this._per_input_losses[inputs_hash] = new List<Tensor>();
                this._per_input_losses[inputs_hash].AddRange(losses);
            }
        }

        private string _object_list_uid(Tensor[] inputs)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        ///   Add updates to the layer.
        /// </summary>
        /// <remarks>
        ///   The updates may potentially be conditional on some inputs tensors, for instance batch norm 
        ///   updates are conditional on the layer's inputs.
        /// </remarks>
        /// 
        /// <param name="updates">Update op or list of update ops to add to the layer.</param>
        /// <param name="inputs">Input tensor or list of inputs tensors to mark the updates as 
        ///   conditional on these inputs. If null is passed, the updates are assumed unconditional.</param>
        /// 
        public void add_update(List<List<Tensor>> updates, List<Tensor> inputs = null)
        {
            if (updates == null || updates.Count == 0)
                return;

            // Update this.updates
            if (this._updates != null)
                this._updates.AddRange(updates);

            // Update this._per_input_updates
            if (inputs != null && inputs.Count == 0)
                inputs = null;

            string inputs_hash;
            if (inputs != null)
            {
                inputs_hash = _object_list_uid(inputs);
            }
            else
            {
                // Updates indexed by null are unconditional
                // rather than input-dependent
                inputs_hash = null;
            }

            if (!this._per_input_updates.ContainsKey(inputs_hash))
                this._per_input_updates[inputs_hash] = new List<List<Tensor>>();
            this._per_input_updates[inputs_hash].AddRange(updates);
        }

        private string _object_list_uid(List<Tensor> inputs)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L2706
            return String.Join(", ", inputs.Select(x => str(id(x))));
        }

        public virtual List<List<Tensor>> get_updates_for(List<Tensor> inputs)
        {
            string inputs_hash;
            if (inputs != null)
                inputs_hash = _object_list_uid(inputs);
            else
                inputs_hash = String.Empty;

            if (this._per_input_updates.ContainsKey(inputs_hash))
                return this._per_input_updates[inputs_hash];
            return new List<List<Tensor>>();
        }


        public virtual List<Tensor> get_losses_for(List<Tensor> inputs)
        {
            string inputs_hash = String.Empty;
            if (inputs != null)
                inputs_hash = _object_list_uid(inputs);

            if (this._per_input_losses.ContainsKey(inputs_hash))
                return this._per_input_losses[inputs_hash];
            return new List<Tensor>();
        }

        /// <summary>
        ///   Returns the current weights of the layer.
        /// </summary>
        /// <returns>
        ///   Weights values as a list of numpy arrays.
        /// </returns>
        /// 
        public virtual List<Tensor> weights
        {
            get { return this.trainable_weights.Concat(this.non_trainable_weights).ToList(); }
        }

        /// <summary>
        ///   Sets the weights of the layer, from .NET arrays.
        /// </summary>
        /// 
        /// <param name="value">
        ///   A list of Numpy arrays. The number of arrays and their shape must match the number of the dimensions of the weights of the layer (i.e.it should match the output of `get_weights`).
        /// </param>
        /// 
        public virtual void set_weights(List<Array> value)
        {
            if (this.weights.Count != value.Count)
                throw new Exception($"You called `set_weights(weights)` on layer {this.name} with a  weight list of length {value.Count}, but the layer was expecting {this.weights.Count} weights. Provided weights: { value }");

            if (this.weights.Count == 0)
                return;

            var weight_value_tuples = new List<Tuple<Tensor, Array>>();

            List<Array> param_values = K.batch_get_value(this.weights);

            for (int i = 0; i < param_values.Count; i++)
            {
                Array pv = param_values[i];
                Tensor p = this.weights[i];
                Array w = value[i];
                if (pv.GetLength().IsEqual(w.GetLength()))
                    throw new Exception($"Layer weight shape {pv.GetLength()} not compatible with provided weight shape {w.GetLength()}");
                weight_value_tuples.Add(Tuple.Create(p, w));
            }

            K.batch_set_value(weight_value_tuples);
        }

        public override string ToString()
        {
            return $"{this.name} ({str(this.input_shape)} -> {str(this.output_shape)})";
        }
    }
}

