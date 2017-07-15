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
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    using static KerasSharp.Backends.Current;
    using static KerasSharp.Engine.Topology.TensorFlowSharp;


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

        private List<Layer> layers;
        public bool _trainable;
        public object _initial_weights;

        public Model model;


        public Sequential(List<Layer> layers = null, string name = null)
        {
            this.layers = new List<Layer>();
            this.model = new Model();
            this.inputs = new List<Tensor>();
            this.outputs = new List<Tensor>();
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

                    // This will build the current layer and create the node connecting 
                    // the current layer to the input layer we just created.
                    layer.Call(x);
                }


                if (layer.inbound_nodes.Count != 1)
                {
                    throw new Exception($"A layer added to a Sequential model must not already be connected somewhere else. Model received layer ' + layer.name which has {layer.inbound_nodes.Count} pre-existing inbound connections.");
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

                // update self.inbound_nodes
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
                // update self.inbound_nodes
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
            this.model.Trainable = this.trainable;

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
    }
}