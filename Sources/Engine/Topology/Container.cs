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
    using KerasSharp.Engine.Topology;
    using System.Collections.Generic;
    using KerasSharp.Constraints;
    using KerasSharp.Layers;
    using System;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    using static KerasSharp.Backends.Current;
    using static KerasSharp.Python;
    using Accord.Math;
    using KerasSharp.Regularizers;

    /// <summary>
    ///   A Container is a directed acyclic graph of layers.
    /// </summary>
    /// 
    /// <remarks>
    ///  It is the topological form of a "model". A Mode  l
    ///  is simply a Container with added training routines.
    /// </remarks>
    /// 
    /// <seealso cref="KerasSharp.Engine.Topology.Layer" />
    /// 
    public class Container : Layer
    {
        protected List<Tensor> inputs;
        protected List<Tensor> masks;
        protected List<Tensor> outputs;
        public List<Layer> input_layers;
        public List<int> input_layers_node_indices;
        public List<int> input_layers_tensor_indices;
        public List<Layer> output_layers;
        public List<int> output_layers_node_indices;
        public List<int> output_layers_tensor_indices;
        protected internal Dictionary<string, List<Tensor>> _output_tensor_cache;
        protected internal Dictionary<string, List<int?[]>> _output_shape_cache;
        protected internal Dictionary<string, List<Tensor>> _output_mask_cache;
        public List<string> input_names;
        public List<string> output_names;
        protected internal List<string> _feed_input_names;
        protected internal List<Tensor> _feed_inputs;
        protected internal List<int?[]> _feed_input_shapes;
        protected int?[][] internal_input_shapes;
        protected int?[][] internal_output_shapes;
        public List<Layer> layers;
        public Dictionary<int, List<Layer>> layers_by_depth;
        public HashSet<string> container_nodes;
        public Dictionary<int, List<Node>> nodes_by_depth;


        public Container()
        {

        }

        public Container(List<Tensor> inputs, List<Tensor> outputs, string name = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L1478

            // Handle `name` argument.
            if (name == null)
            {
                string prefix = this.GetType().Name.ToLowerInvariant();
                name = prefix + '_' + K.get_uid(prefix);
            }
            this.name = name;

            this.supports_masking = false;
            this.trainable = true;

            this.inputs = inputs;
            this.outputs = outputs;

            // Check for redundancy in inputs.
            if (this.inputs.Count != this.inputs.Distinct().Count())
                throw new Exception("The list of inputs passed to the model is redundant. All inputs should only appear once. Found: " + this.inputs);

            // Check for redundancy in outputs.
            if (this.outputs.Count != this.outputs.Distinct().Count())
                Trace.TraceWarning("The list of outputs passed to the model is redundant. All outputs should only appear once. Found: " + this.outputs);

            // List of initial layers (1 to 1 mapping with this.inputs,
            // hence the same layer might appear twice)
            this.input_layers = new List<Layer>();
            this.input_layers_node_indices = new List<int>();
            this.input_layers_tensor_indices = new List<int>();

            // list of layers (1 to 1 mapping with this.inputs,
            // hence the same layer might appear twice)
            this.output_layers = new List<Layer>();
            this.output_layers_node_indices = new List<int>();
            this.output_layers_tensor_indices = new List<int>();

            // all layers in order of horizontal graph traversal.
            // Entries are unique. Includes input and output layers.
            this.layers = new List<Layer>();

            // This is for performance optimization when calling the Container on new inputs.
            // every time the Container is called on a set on input tensors, we compute the output tensors,
            // output masks and output shapes in one pass, then cache them here. When of of these output is 
            // queried later, we retrieve it from there instead of recomputing it.
            this._output_mask_cache = new Dictionary<string, List<Tensor>>();
            this._output_tensor_cache = new Dictionary<string, List<Tensor>>();
            this._output_shape_cache = new Dictionary<string, List<int?[]>>();

            // User-provided arguments validation.
            foreach (Tensor x in this.inputs)
            {
                // Check that x is an input tensor.
                var (layer, node_index, tensor_index) = x._keras_history.Value;

                if (layer.inbound_nodes.Count > 1 || (layer.inbound_nodes != null && layer.inbound_nodes[0].inbound_layers != null))
                {
                    string cls_name = this.GetType().Name;
                    Trace.TraceWarning($"{cls_name} inputs must come from a Keras Input layer, they cannot be the output of a previous non-Input layer. Here, a tensor specified as input to {this.name} was not an Input tensor, it was generated by layer {layer.name}. Note that input tensors are instantiated via 'tensor = Input(shape)'. The tensor that caused the issue was: {x.name}");
                }
            }

            foreach (Tensor x in this.outputs)
            {
                var (layer, node_index, tensor_index) = x._keras_history.Value;
                this.output_layers.Add(layer);
                this.output_layers_node_indices.Add(node_index);
                this.output_layers_tensor_indices.Add(tensor_index);
            }

            // Fill in the output mask cache.
            var masks = new List<Tensor>();
            foreach (Tensor x in this.inputs)
            {
                var (layer, node_index, tensor_index) = x._keras_history.Value;
                Node node = layer.inbound_nodes[node_index];
                Tensor mask = node.output_masks[tensor_index];
                masks.Add(mask);
            }

            string mask_cache_key = String.Join(",", this.inputs.Select(x => str(id(x))));
            mask_cache_key += "_" + String.Join(",", masks.Select(x => str(id(x))));

            masks = new List<Tensor>();
            foreach (Tensor x in this.outputs)
            {
                var (layer, node_index, tensor_index) = x._keras_history.Value;
                Node node = layer.inbound_nodes[node_index];
                Tensor mask = node.output_masks[tensor_index];
                masks.Add(mask);
            }


            this._output_mask_cache[mask_cache_key] = masks;

            // Build this.input_layers:
            foreach (Tensor x in this.inputs)
            {
                var (layer, node_index, tensor_index) = x._keras_history.Value;
                // It's supposed to be an input layer, so only one node
                // and one tensor output.
                Trace.Assert(node_index == 0);
                Trace.Assert(tensor_index == 0);
                this.input_layers.Add(layer);
                this.input_layers_node_indices.Add(node_index);
                this.input_layers_tensor_indices.Add(tensor_index);
            }

            // Build this.input_names and this.output_names.
            this.input_names = new List<string>();
            this.output_names = new List<string>();
            this._feed_input_names = new List<string>();
            this._feed_inputs = new List<Tensor>();
            this._feed_input_shapes = new List<int?[]>();

            for (int i = 0; i < this.input_layers.Count; i++)
            {
                Layer layer = this.input_layers[i];

                // Check that layer is an InputLayer.
                if (!(layer is InputLayer))
                    throw new Exception($"Input layers to a 'Model' must be 'InputLayer' objects. Received inputs: {inputs}. Input {i} (0-based) originates from layer type '{layer}'.");

                this.input_names.Add(layer.name);

                if (layer.is_placeholder)
                {
                    this._feed_input_names.Add(layer.name);
                    this._feed_inputs.AddRange(layer.input);
                    this._feed_input_shapes.Add(this.inputs[i]._keras_shape);
                }
            }

            foreach (var layer in this.output_layers)
                this.output_names.Add(layer.name);

            this.internal_input_shapes = this.inputs.Select(x => x._keras_shape).ToArray();
            this.internal_output_shapes = this.outputs.Select(x => x._keras_shape).ToArray();

            // Container_nodes: set of nodes included in the graph
            // (not all nodes included in the layers
            // are relevant to the current graph).
            var container_nodes = new HashSet<string>();  // ids of all nodes relevant to the Container
            var nodes_depths = new Dictionary<Node, int>();  // dict {node: depth value}
            var layers_depths = new Dictionary<Layer, int>();  // dict {layer: depth value}
            var layer_indices = new Dictionary<Layer, int>();  // dict {layer: index in traversal}
            List<Node> nodes_in_decreasing_depth = new List<Node>();

            /// <summary>
            ///   Builds a map of the graph of layers.
            /// </summary>
            /// <remarks>
            ///   This recursively updates the map `layer_indices`, the list `nodes_in_decreasing_depth` and the set `container_nodes`.
            /// </remarks>
            /// 
            /// <param name="finished_nodes">Set of nodes whose subgraphs have been traversed completely.Useful to prevent duplicated work.</param>
            /// <param name="nodes_in_progress">Set of nodes that are currently active on the recursion stack.Useful to detect cycles.</param>
            /// <param name="layer">Layer from which `tensor` comes from. If not provided, will be obtained from `tensor._keras_history`.</param>
            /// <param name="node_index">Node index from which `tensor` comes from.</param>
            /// <param name="tensor_index">Tensor_index from which `tensor` comes from.</param>
            ///			
            void build_map_of_graph(Tensor tensor, HashSet<Node> finished_nodes = null, HashSet<Node> nodes_in_progress = null,
                                       Layer layer = null, int? node_index = null, int? tensor_index = null)
            {
                if (layer == null || node_index == null || tensor_index == null)
                    (layer, node_index, tensor_index) = tensor._keras_history.Value;

                Node node = layer.inbound_nodes[node_index.Value];

                // Prevent cycles.
                if (nodes_in_progress.Contains(node))
                    throw new Exception("The tensor ' + str(tensor) + ' at layer " + layer.name + " is part of a cycle.");

                // Don't repeat work for shared subgraphs
                if (finished_nodes.Contains(node))
                    return;

                string node_key = layer.name + "_ib-" + node_index;

                // Update container_nodes.
                container_nodes.Add(node_key);

                // Store the traversal order for layer sorting.
                if (!layer_indices.ContainsKey(layer))
                    layer_indices[layer] = layer_indices.Count;

                nodes_in_progress.Add(node);

                // Propagate to all previous tensors connected to this node.
                for (int i = 0; i < node.inbound_layers.Count; i++)
                {
                    Tensor x = node.input_tensors[i];
                    layer = node.inbound_layers[i];
                    node_index = node.node_indices[i];
                    tensor_index = node.tensor_indices[i];
                    build_map_of_graph(x, finished_nodes, nodes_in_progress, layer, node_index, tensor_index);
                }

                finished_nodes.Add(node);
                nodes_in_progress.Remove(node);

                nodes_in_decreasing_depth.Add(node);
                return;
            }

            {
                var finished_nodes = new HashSet<Node>();
                var nodes_in_progress = new HashSet<Node>();
                foreach (var x in this.outputs)
                    build_map_of_graph(x, finished_nodes, nodes_in_progress);

                int depth = 0;
                foreach (Node node in Enumerable.Reverse(nodes_in_decreasing_depth))
                {
                    // If the depth is not set, the node has no outbound nodes (depth 0).
                    if (!nodes_depths.ContainsKey(node))
                        nodes_depths[node] = 0;
                    depth = nodes_depths[node];

                    // Update the depth of the corresponding layer
                    int previous_depth = 0;
                    if (layers_depths.ContainsKey(node.outbound_layer))
                        previous_depth = layers_depths[node.outbound_layer];

                    // If we've seen this layer before at a higher depth, we should use that depth instead
                    // of the node depth.  This is necessary for shared layers that have inputs at different
                    // depth levels in the graph.
                    depth = Math.Max(depth, previous_depth);
                    layers_depths[node.outbound_layer] = depth;
                    nodes_depths[node] = depth;

                    // Update the depth of inbound nodes.
                    for (int i = 0; i < node.inbound_layers.Count; i++)
                    {
                        var inbound_layer = node.inbound_layers[i];
                        int node_index = node.node_indices[i].Value;
                        var inbound_node = inbound_layer.inbound_nodes[node_index];
                        previous_depth = 0;
                        nodes_depths.TryGetValue(inbound_node, out previous_depth);
                        nodes_depths[inbound_node] = Math.Max(depth + 1, previous_depth);
                    }
                }

                // Build a dict {depth: list of nodes with this depth}
                var nodes_by_depth = new Dictionary<int, List<Node>>();
                foreach (Node node in nodes_depths.Keys)
                {
                    depth = nodes_depths[node];
                    if (!nodes_by_depth.ContainsKey(depth))
                        nodes_by_depth[depth] = new List<Node>();
                    nodes_by_depth[depth].Add(node);
                }

                // Build a dict {depth: list of layers with this depth}
                var layers_by_depth = new Dictionary<int, List<Layer>>();
                foreach (Layer layer in layers_depths.Keys)
                {
                    depth = layers_depths[layer];
                    if (!layers_by_depth.ContainsKey(depth))
                        layers_by_depth[depth] = new List<Layer>();
                    layers_by_depth[depth].Add(layer);
                }

                // Get sorted list of layer depths.
                var depth_keys = layers_by_depth.Keys.ToList();
                depth_keys.Sort();
                depth_keys.Reverse();

                // Set this.layers and this.layers_by_depth.
                var layers = new List<Layer>();
                foreach (int d in depth_keys)
                {
                    var layers_for_depth = layers_by_depth[d];

                    // Container.layers needs to have a deterministic order:
                    // here we order them by traversal order.
                    layers_for_depth = layers_for_depth.OrderBy(x => layer_indices[x]).ToList();
                    foreach (var l in layers_for_depth)
                        layers.Add(l);
                }

                this.layers = layers;
                this.layers_by_depth = layers_by_depth;

                // Get sorted list of node depths.
                depth_keys = nodes_by_depth.Keys.ToList();
                depth_keys.Sort();
                depth_keys.Reverse();

                // Check that all tensors required are computable.
                // computable_tensors: all tensors in the graph
                // that can be computed from the inputs provided.
                var computable_tensors = new List<Tensor>();
                foreach (var x in this.inputs)
                    computable_tensors.Add(x);


                var layers_with_complete_input = new List<string>();  // To provide a better error msg.
                foreach (int? d in depth_keys)
                {
                    foreach (Node node in nodes_by_depth[d.Value])
                    {
                        var layer = node.outbound_layer;

                        if (layer != null)
                        {
                            foreach (Tensor x in node.input_tensors)
                            {
                                if (!computable_tensors.Contains(x))
                                    throw new Exception($"Graph disconnected: cannot obtain value for tensor {x} at layer {layer.name}. The following previous layers were accessed without issue: {layers_with_complete_input}");
                            }

                            foreach (var x in node.output_tensors)
                                computable_tensors.Add(x);

                            layers_with_complete_input.Add(layer.name);
                        }
                    }
                }

                // Set this.nodes and this.nodes_by_depth.
                this.container_nodes = container_nodes;
                this.nodes_by_depth = nodes_by_depth;

                // Ensure name unicity, which will be crucial for serialization
                // (since serialized nodes refer to layers by their name).
                var all_names = this.layers.Select(x => x.name);
                foreach (string n in all_names)
                {
                    int count = all_names.Count(x => x == n);
                    if (count != 1)
                        throw new Exception($"The name {name} is used {count} times in the model. All layer names should be unique.");
                }

                // Layer parameters.
                // The new container starts with a single inbound node
                // for its inputs, and no outbound nodes.
                this.outbound_nodes = new List<Node>();  // Will be appended to by future calls to __call__
                this.inbound_nodes = new List<Node>();  // Will be appended to below, and by future calls to __call__

                // Create the node linking internal inputs to internal outputs.
                new Node(outbound_layer: this,
                         inbound_layers: new List<Layer>(),
                         node_indices: new List<int?>(),
                         tensor_indices: new List<int?>(),
                         input_tensors: this.inputs,
                         output_tensors: this.outputs,
                         // No container-level masking for now.
                         input_masks: this.inputs.Select(x => (Tensor)null).ToList(),
                         output_masks: this.outputs.Select(x => (Tensor)null).ToList(),
                         input_shapes: this.inputs.Select(x => x._keras_shape).ToList(),
                         output_shapes: this.outputs.Select(x => x._keras_shape).ToList());

                this.built = true;
            }
        }


        /// <summary>
        ///   Retrieves a layer based on either its name (unique) or index.
        /// </summary>
        ///
        /// <remarks>
        ///   Indices are based on order of horizontal graph traversal(bottom-up).
        /// </remarks>
        /// name: String, name of layer.
        /// index: Integer, index of layer.
        /// <returns>A layer instance.</returns>
        ///
        public virtual Layer get_layer(string name = null, int? index = null)
        {
            // It would be unreliable to build a dictionary
            // based on layer names, because names can potentially
            // be changed at any point by the user
            // without the container being notified of it.
            if (index != null)
            {
                if (this.layers.Count <= index)
                {
                    throw new Exception("Was asked to retrieve layer at index {index} but model only has {this.layers.Length} layers.");
                }
                else
                {
                    return this.layers[index.Value];
                }
            }
            else
            {
                if (name == null)
                    throw new Exception("Provide either a layer name or layer index.");
            }

            foreach (Layer layer in this.layers)
            {
                if (layer.name == name)
                    return layer;
            }

            throw new Exception("No such layer: " + name);
        }

        /// <summary>
        ///   Returns the `updates` from all layers that are stateful.
        /// </summary>
        /// 
        /// <returns>A list of update ops.</returns>
        /// 
        /// <remarks>Will only include updates that are either inconditional, or conditional on inputs to this model
        /// (e.g.will not include updates that depend on tensors that aren't inputs to this model).</remarks>
        /// 
        public override List<Tensor> updates
        {
            get
            {
                // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L1874
                var updates = new List<Tensor>();
                foreach (Layer layer in this.layers)
                {
                    if (layer.updates != null)
                    {
                        // Collect updates that are dependent on inputs
                        // that are part of the model.
                        for (int node_index = 0; node_index < layer.inbound_nodes.Count; node_index++)
                        {
                            Node node = layer.inbound_nodes[node_index];

                            string node_key = $"{layer.name}_ib-{node_index}";

                            if (this.container_nodes.Contains(node_key))
                            {
                                // The model owns this layer node.
                                inputs = node.input_tensors;
                                updates.AddRange(layer.get_updates_for(inputs));
                            }
                        }
                        // Collect unconditional updates.
                        updates.AddRange(layer.get_updates_for(null));
                    }
                }

                return updates;
            }
        }


        /// <summary>
        ///   Retrieve the model's losses.
        /// </summary>
        /// 
        /// <remarks>
        ///   Will only include losses that are either inconditional, or conditional on inputs to this model (e.g.will not include losses that depend on tensors that aren't inputs to this model).
        /// </remarks>
        /// 
        /// <returns>A list of loss tensors.</returns>
        ///
        public override List<Tensor> losses
        {
            get
            {
                var losses = new List<Tensor>();
                // Retrieve losses for all internal layers.
                foreach (var layer in this.layers)
                {
                    if (layer.losses != null)
                    {
                        // Collect losses that are dependent on inputs
                        // that are part of the model.
                        for (int node_index = 0; node_index < layer.inbound_nodes.Count; node_index++)
                        {
                            Node node = layer.inbound_nodes[node_index];

                            string node_key = $"{layer.name}_ib-{node_index}";

                            if (this.container_nodes.Contains(node_key))
                            {
                                // The model owns this layer node.
                                inputs = node.input_tensors;
                                losses.AddRange(layer.get_losses_for(inputs));
                            }
                        }

                        // Collect unconditional losses.
                        losses.AddRange(layer.get_losses_for(null));
                    }
                }

                // Add any potential unconditional model-level loss.
                losses.AddRange(this.get_losses_for(null));
                return losses;
            }
        }


        public override bool uses_learning_phase
        {
            get
            {
                foreach (var x in this.outputs)
                    if (x._uses_learning_phase)
                        return true;
                return false;
            }
        }

        public override bool stateful
        {
            get
            {
                foreach (var x in this.layers)
                    if (x.stateful)
                        return true;
                return false;
            }
        }

        public override void reset_states()
        {
            foreach (var layer in this.layers)
                layer.reset_states();
        }

        /// <summary>
        ///   Updates model states.
        /// </summary>
        /// 
        /// <remarks>
        ///   This is useful for separating training updates and state updates, e.g.when we need to update a layer's internal state during prediction.
        /// </remarks>
        /// 
        /// <returns>A list of update ops.</returns>
        /// 
        public virtual List<Tensor> state_updates()
        {
            var state_updates = new List<Tensor>(); ;
            foreach (Layer layer in this.layers)
            {
                if (layer.stateful == false)
                    if (layer.updates != null)
                        state_updates.AddRange(layer.updates);
            }
            return state_updates;
        }


        public virtual Dictionary<Tensor, IWeightRegularizer> regularizers
        {
            get; set;
        }

        public override Dictionary<Tensor, IWeightConstraint> constraints
        {
            get
            {
                var cons = new Dictionary<Tensor, IWeightConstraint>();
                foreach (Layer layer in this.layers)
                {
                    foreach (Tensor key in layer.constraints.Keys)
                    {
                        var value = layer.constraints[key];
                        if (cons.ContainsKey(key) && cons[key] != value)
                            throw new Exception($"Received multiple constraints for one weight tensor: {key}");
                        cons[key] = value;
                    }
                }
                return cons;
            }
        }


        public override List<Tensor> trainable_weights
        {
            get
            {
                var weights = new List<Tensor>();
                if (!this.trainable)
                    return weights;

                foreach (var layer in this.layers)
                    weights.AddRange(layer.trainable_weights);
                return weights;
            }
        }

        public override List<Tensor> non_trainable_weights
        {
            get
            {
                var weights = new List<Tensor>();
                foreach (Layer layer in this.layers)
                    weights.AddRange(layer.non_trainable_weights);

                if (!this.trainable)
                {
                    var trainable_weights = new List<Tensor>();
                    foreach (Layer layer in this.layers)
                        trainable_weights.AddRange(layer.trainable_weights);

                    trainable_weights.AddRange(weights);
                    return trainable_weights;
                }

                return weights;
            }
        }

        /// <summary>
        ///   Retrieves the weights of the model.
        /// </summary>
        /// 
        /// <remarks>
        ///   A flat list of Numpy arrays.
        /// </remarks>
        /// 
        public List<Array> get_weights()
        {
            var weights = new List<List<Tensor>>();
            foreach (var layer in this.layers)
                weights.Add(layer.weights);
            return K.batch_get_value(weights);
        }

        /// <summary>
        ///   Sets the weights of the model.
        /// </summary>
        /// 
        /// <param name="weights">A list of Numpy arrays with shapes and types matching the output of `model.get_weights()`.</param>
        /// 
        public override void set_weights(List<Array> weights)
        {
            var tuples = new List<(Tensor, Array)>();
            foreach (var layer in this.layers)
            {
                int num_param = layer.weights.Count;
                List<Array> layer_weights = weights.ToArray().Get(0, num_param).ToList();
                for (int i = 0; i < layer.weights.Count; i++)
                {
                    var sw = layer.weights[i];
                    var w = layer_weights[i];
                    tuples.Add((sw, w));
                }
                weights = weights.ToArray().Get(num_param, 0).ToList();
            }
            K.batch_set_value(tuples);
        }

        /// <summary>
        ///   Gets the model's input specs.
        /// </summary>
        /// 
        /// <remarks>
        ///   A list of `InputSpec` instances (one per input to the model) or a single instance if the model has only one input.
        /// </remarks>
        /// 
        public override List<InputSpec> input_spec
        {
            get
            {
                var specs = new List<InputSpec>();
                foreach (Layer layer in this.input_layers)
                {
                    if (layer.input_spec == null)
                    {
                        specs.Add(null);
                    }
                    else
                    {
                        specs.AddRange(layer.input_spec);
                    }
                }

                return specs;
            }
        }

        /// <summary>
        ///   Call the model on new inputs.
        /// </summary>
        /// 
        /// <remarks>
        ///   In this case `call` just reapplies all ops in the graph to the new inputs (e.g.build a new computational graph 
        ///   from the provided inputs). A model is callable on non - Keras tensors.
        /// </remarks>
        /// 
        /// <param name="inputs">A tensor or list of tensors..</param>
        /// <param name="masks">A mask or list of masks. A mask can be either a tensor or null (no mask).</param>
        /// 
        /// <returns>A tensor if there is a single output, or a list of tensors if there are more than one outputs.</returns>
        /// 
        public List<Tensor> call(List<Tensor> inputs, List<Tensor> mask = null)
        {
            if (mask == null)
                masks = inputs.Select(x => (Tensor)x).ToList();

            string cache_key = String.Join(",", inputs.Select(x => str(id(x))));
            cache_key += '_' + String.Join(",", masks.Select(x => str(id(x))));

            if (this._output_tensor_cache.ContainsKey(cache_key))
                return this._output_tensor_cache[cache_key];

            var (output_tensors, _, _) = this.run_internal_graph(inputs, masks);
            return output_tensors;
        }

        public override List<Tensor> compute_mask(List<Tensor> inputs, List<Tensor> mask)
        {
            if (mask == null)
                masks = inputs.Select(x => (Tensor)null).ToList();

            string cache_key = String.Join(",", inputs.Select(x => str(id(x))));
            cache_key += '_' + String.Join(",", masks.Select(x => str(id(x))));

            if (this._output_mask_cache.ContainsKey(cache_key))
                return this._output_mask_cache[cache_key];

            var (_, output_masks, _) = this.run_internal_graph(inputs, masks);
            return output_masks;
        }

        public override List<int?[]> compute_output_shape(List<int?[]> input_shapes)
        {
            if (input_shapes.Count != this.input_layers.Count)
                throw new Exception($"Invalid input_shape argument {input_shape}: model has {this.input_layers.Count} tensor inputs.");

            string cache_key = String.Join(",", input_shapes.Select(x => str(x)));
            if (this._output_shape_cache.ContainsKey(cache_key))
                return this._output_shape_cache[cache_key];

            // Bad luck, we have to run the graph manually.
            var layers_to_output_shapes = new Dictionary<string, int?[]>();
            for (int i = 0; i < input_shapes.Count; i++)
            {
                var layer = this.input_layers[i];
                var input_shape = input_shapes[i];
                // It's an input layer: compute_output_shape is identity,
                // and there is only one node and one tensor output.
                string shape_key = layer.name + "_0_0";
                layers_to_output_shapes[shape_key] = input_shape;
            }

            var depth_keys = this.nodes_by_depth.Keys.ToList();
            depth_keys.Sort();
            depth_keys.Reverse();

            // Iterate over nodes, by depth level.
            if (depth_keys.Count > 1)
            {
                foreach (var depth in depth_keys)
                {
                    var nodes = this.nodes_by_depth[depth];
                    foreach (var node in nodes)
                    {
                        // This is always a single layer, never a list.
                        var layer = node.outbound_layer;
                        if (this.input_layers.Contains(layer))
                        {
                            // We've already covered the input layers
                            // a few lines above.
                            continue;
                        }

                        // Potentially redundant list,
                        // same size of node.input_tensors.
                        input_shapes = new List<int?[]>();
                        for (int j = 0; j < node.inbound_layers.Count; j++)
                        {
                            var inbound_layer = node.inbound_layers[j];
                            int node_index = node.node_indices[j].Value;
                            var tensor_index = node.tensor_indices[j];
                            var shape_key = inbound_layer.name + $"_{node_index}_{tensor_index}";
                            var input_shape = layers_to_output_shapes[shape_key];
                            input_shapes.Add(input_shape);
                        }


                        List<int?[]> output_shapes = layer.compute_output_shape(input_shapes);
                        {
                            int node_index = layer.inbound_nodes.FindIndex(x => x == node);
                            for (int j = 0; j < output_shapes.Count; j++)
                            {
                                string shape_key = layer.name + $"_{node_index}_{j}";
                                layers_to_output_shapes[shape_key] = output_shapes[j];
                            }
                        }
                    }
                }
            }

            {
                // Read final output shapes from layers_to_output_shapes.
                var output_shapes = new List<int?[]>();
                var output_shape_keys = new List<string>();
                for (int i = 0; i < this.output_layers.Count; i++)
                {
                    Layer layer = this.output_layers[i];
                    int node_index = this.output_layers_node_indices[i];
                    var tensor_index = this.output_layers_tensor_indices[i];
                    string shape_key = layer.name + $"_{node_index}_{tensor_index}";
                    output_shape_keys.Add(shape_key);
                }

                for (int i = 0; i < output_shape_keys.Count; i++)
                {
                    var key = output_shape_keys[i];
                    Trace.Assert(layers_to_output_shapes.ContainsKey(key));
                    output_shapes.Add(layers_to_output_shapes[key]);
                }

                // Store in cache.
                this._output_shape_cache[cache_key] = output_shapes;
                return output_shapes;
            }
        }

        /// <summary>
        ///   Computes output tensors for new inputs.
        /// </summary>
        ///
        /// <remarks>
        ///   - Expects `inputs` to be a list(potentially with 1 element).
        ///   - Can be run on non-Keras tensors.
        /// </remarks>
        ///
        /// <param name="inputs">List of tensors</param>
        /// <param name="masks">List of masks</param>
        ///
        /// <returns>Three lists: output_tensors, output_masks, output_shapes</returns>
        ///
        public (List<Tensor>, List<Tensor>, List<int?[]>) run_internal_graph(List<Tensor> inputs, List<Tensor> masks = null)
        {
            if (masks == null)
                masks = inputs.Select(x => (Tensor)null).ToList();

            // Dictionary mapping reference tensors to tuples
            // (computed tensor, compute mask)
            // we assume a 1:1 mapping from tensor to mask
            // TODO: raise exception when a '.compute_mask()' call
            // does not return a list the same size as 'call'
            var tensor_map = new Dictionary<long, (Tensor, Tensor)>();

            for (int i = 0; i < this.inputs.Count; i++)
            {
                Tensor x = this.inputs[i];
                Tensor y = inputs[i];
                Tensor mask = masks[i];

                tensor_map[id(x)] = (y, mask);

                var depth_keys = this.nodes_by_depth.Keys.ToList();
                depth_keys.Sort();
                depth_keys.Reverse();

                foreach (var depth in depth_keys)
                {
                    List<Node> nodes = this.nodes_by_depth[depth];
                    foreach (Node node in nodes)
                    {
                        // This is always a single layer, never a list.
                        var layer = node.outbound_layer;
                        var reference_input_tensors = node.input_tensors;
                        var reference_output_tensors = node.output_tensors;

                        // If all previous input tensors are available in tensor_map,
                        // then call node.inbound_layer on them.
                        var computed_data = new List<(Tensor, Tensor)>();  // List of tuples (input, mask).
                        foreach (Tensor t in reference_input_tensors)
                        {
                            if (tensor_map.ContainsKey(id(t)))
                                computed_data.Add(tensor_map[id(t)]);
                        }

                        if (computed_data.Count == reference_input_tensors.Count)
                        {
                            // call layer
                            using (K.name_scope(layer.name))
                            {
                                List<Tensor> innerMask = null;
                                List<Tensor> computed_tensors = null;
                                List<Tensor> computed_masks;
                                List<Tensor> output_tensors;
                                List<Tensor> output_masks;
                                if (computed_data.Count == 1)
                                {
                                    var (computed_tensor, computed_mask) = computed_data[0];
                                    output_tensors = layer.Call(computed_tensor);
                                    output_masks = layer.compute_mask(computed_tensor, computed_mask);
                                    computed_tensors = new List<Tensor>() { computed_tensor };
                                    computed_masks = new List<Tensor> { computed_mask };
                                }
                                else
                                {
                                    computed_tensors = computed_data.Select(t => t.Item1).ToList();
                                    computed_masks = computed_data.Select(t => t.Item2).ToList();
                                    innerMask = computed_masks;
                                    output_tensors = layer.Call(computed_tensors, mask: innerMask);
                                    output_masks = layer.compute_mask(computed_tensors, computed_masks);
                                }

                                // Apply activity regularizer if any:
                                if (layer.activity_regularizer != null)
                                {
                                    var regularization_losses = computed_tensors.Select(t => layer.activity_regularizer.Call(t)).ToList();
                                    layer.add_loss(regularization_losses, computed_tensors);
                                }

                                // Update model updates and losses:
                                // Keep track of updates that depend on the inputs
                                // (e.g. BN updates).
                                this.add_update(layer.get_updates_for(computed_tensors), inputs);

                                // Keep track of unconditional updates (e.g. a counter).
                                this.add_update(layer.get_updates_for(null), null);

                                // Keep track of losses that depend on the inputs
                                // (e.g. activity regularizers).
                                this.add_loss(layer.get_losses_for(computed_tensors), inputs);

                                // Keep track of unconditional losses
                                // (e.g. weight regularizers).
                                this.add_loss(layer.get_losses_for(null), null);


                                // Update _keras_shape.
                                if (computed_tensors.Any(e => e._keras_shape != null))
                                {
                                    var shapes = layer.compute_output_shape(computed_tensors.Select(t => t._keras_shape).ToList());
                                    uses_learning_phase = computed_tensors.Any(t => t._uses_learning_phase);

                                    for (int j = 0; j < output_tensors.Count; j++)
                                    {
                                        var t = output_tensors[j];
                                        var s = shapes[j];
                                        t._keras_shape = s;
                                        t._uses_learning_phase = t._uses_learning_phase || uses_learning_phase;
                                    }
                                }

                                // Update tensor_map.
                                for (int j = 0; j < reference_output_tensors.Count; j++)
                                {
                                    var xt = reference_output_tensors[j];
                                    var yt = output_tensors[j];
                                    var mt = output_masks[j];
                                    tensor_map[id(xt)] = (yt, mt);
                                }
                            }
                        }
                    }
                }
            }

            {
                List<Tensor> output_tensors = new List<Tensor>();
                List<Tensor> output_masks = new List<Tensor>();
                List<int?[]> output_shapes = new List<int?[]>();

                foreach (Tensor x in this.outputs)
                {
                    Trace.Assert(tensor_map.ContainsKey(id(x)), $"Could not compute output {x}");
                    var (tensor, mask) = tensor_map[id(x)];
                    if (tensor._keras_shape != null && output_shapes != null)
                    {
                        var shape = tensor._keras_shape;
                        output_shapes.Add(shape);
                    }
                    else
                    {
                        output_shapes = null;
                    }

                    output_tensors.Add(tensor);
                    output_masks.Add(mask);
                }

                // Update cache;
                // keys are based on ids on input tensors and inputs masks.
                string cache_key = String.Join(",", inputs.Select(x => str(id(x))));
                cache_key += "_" + String.Join(",", masks.Select(x => str(id(x))));

                this._output_tensor_cache[cache_key] = output_tensors;
                this._output_mask_cache[cache_key] = output_masks;

                if (output_shapes != null)
                {
                    var input_shapes = inputs.Select(x => x._keras_shape).ToList();
                    cache_key = String.Join(",", input_shapes.Select(x => str(x)));
                    this._output_shape_cache[cache_key] = output_shapes;
                }
                else
                {
                    this._output_shape_cache[cache_key] = output_shapes;
                }

                return (output_tensors, output_masks, output_shapes);
            }
        }

        /// <summary>
        ///   Returns the list of input tensors necessary to compute `tensor`.
        /// </summary>
        ///
        /// <remarks>
        ///   Output will always be a list of tensors (potentially with 1 element).
        /// </remarks>
        ///
        /// <returns>List of input tensors.</returns>
        ///
        public List<Tensor> get_source_inputs(Tensor tensor, Layer layer = null, int? node_index = null)
        {
            //		tensor: The tensor to start from.
            //		layer: Origin layer of the tensor.Will be
            //			determined via tensor._keras_history if not provided.
            //		node_index: Origin node index of the tensor.


            if (layer == null || node_index == null)
                (layer, node_index, _) = tensor._keras_history.Value;

            if (layer.inbound_nodes.Count == 0)
                return new List<Tensor>() { tensor };


            var node = layer.inbound_nodes[node_index.Value];

            if (node.inbound_layers.Count == 0)
            {
                // Reached an Input layer, stop recursion.
                return node.input_tensors;
            }

            var source_tensors = new List<Tensor>();

            for (int i = 0; i < node.inbound_layers.Count; i++)
            {
                var x = node.input_tensors[i];
                layer = node.inbound_layers[i];
                node_index = node.node_indices[i];
                List<Tensor> previous_sources = get_source_inputs(x, layer, node_index);

                // Avoid input redundancy.
                foreach (Tensor t in previous_sources)
                {
                    if (!source_tensors.Contains(t))
                        source_tensors.Add(t);
                }
            }

            return source_tensors;
        }





        public static string _object_list_uid(List<Tensor> object_list)
        {
            return string.Join(", ", object_list.Select(x => str(id(x))));
        }




        /// <summary>
        ///   Retrieves the output mask(s) of the previous node.
        /// </summary>
        ///
        /// <param name="input_tensors">A tensor or list of tensors.</param>
        ///
        /// <returns>A mask tensor or list of mask tensors.</returns>
        ///
        public static List<Tensor> _collect_previous_mask(List<Tensor> input_tensors)
        {
            var masks = new List<Tensor>();

            foreach (Tensor x in input_tensors)
            {
                if (x._keras_history.HasValue)
                {
                    var (inbound_layer, node_index, tensor_index) = x._keras_history.Value;
                    var node = inbound_layer.inbound_nodes[node_index];
                    var mask = node.output_masks[tensor_index];
                    masks.Add(mask);
                }
                else
                {
                    masks.Add(null);
                }
            }
            return masks;
        }

        /// <summary>
        ///   Collects the output shape(s) of a list of Keras tensors.
        /// </summary>
        /// 
        /// <param name="input_tensors">List of input tensors(or single input tensor).</param>
        /// 
        /// <returns>List of shape tuples(or single tuple), one tuple per input.</returns>
        /// 
        public static List<int?[]> _collect_input_shape(List<Tensor> input_tensors)
        {
            var shapes = new List<int?[]>();

            foreach (Tensor x in input_tensors)
            {
                try
                {
                    shapes.Add(K.int_shape(x));
                }
                catch
                {
                    shapes.Add(null);
                }
            }
            return shapes;
        }
    }
}
