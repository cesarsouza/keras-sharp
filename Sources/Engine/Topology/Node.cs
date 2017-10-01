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
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;
    
    using static KerasSharp.Backends.Current;

    using static KerasSharp.Python;


    /// <summary>
    ///   A Node describes the connectivity between two layers.
    /// </summary>
    /// <remarks>
    ///   Each time a layer is connected to some new input, a node is added to
    ///   <see cref="Layer.InboundNodes"/>. Each time the output of a layer is 
    ///   used by another layer, a node is added to <see cref="Layer.OutboundNodes"/>.
    /// </remarks>
    /// 
    [DataContract]
    public class Node
    {
        public Layer outbound_layer;
        public List<Layer> inbound_layers;
        public List<int?> node_indices;
        public List<int?> tensor_indices;
        public List<Tensor> input_tensors;
        public List<Tensor> output_tensors;
        public List<Tensor> input_masks;
        public List<Tensor> output_masks;
        public List<int?[]> input_shapes;
        public List<int?[]> output_shapes;
        public List<Tensor> input_mask;
        public List<Tensor> output_mask;
        private object arguments;

        /// <summary>
        /// Initializes a new instance of the <see cref="Node"/> class.
        /// </summary>
        /// <param name="outbound_layer">The layer that takes <paramref name="input_tensors"/> and turns them <paramref name="output_tensors"/>.
        ///   (the node gets created when the <see cref="Layer.Call"/> method of the layer was called).</param>
        /// <param name="inbound_layers">A list of layers, the same length as <paramref name="input_tensors"/>, the layers from where 
        ///   <paramref name="input_tensors"/> originate.</param>
        /// <param name="node_indices">A list of integers, the same length as <paramref name="inbound_layers"/>. Each
        ///   element in this list denotes the origin node of each tensor in <paramref name="input_tensors"/>.
        /// <param name="tensor_indices">A list of integers, the same length as <paramref name="inbound_layers"/>. Each
        ///   element in this list denotes the index of each tensor in <paramref name="input_tensors"/> within the
        ///   output of their associated <paramref name="inbound_layers"/>.
        /// <param name="input_tensors">The list of input tensors.</param>
        /// <param name="output_tensors">The list of output tensors.</param>
        /// <param name="input_masks">The list of input masks (a mask can be a tensor, or null).</param>
        /// <param name="output_masks">The list of input masks (a mask can be a tensor, or null).</param>
        /// <param name="input_shapes">The list of input shapes.</param>
        /// <param name="output_shapes">The list of output shapes.</param>
        public Node(Layer outbound_layer,
                    List<Layer> inbound_layers, List<int?> node_indices, List<int?> tensor_indices,
                     List<Tensor> input_tensors, List<Tensor> output_tensors,
                     List<Tensor> input_masks, List<Tensor> output_masks,
                     List<int?[]> input_shapes, List<int?[]> output_shapes,
                     object arguments = null)
        {
            // Layer instance (NOT a list).
            // this is the layer that takes a list of input tensors
            // and turns them into a list of output tensors.
            // the current node will be added to
            // the inbound_nodes of outbound_layer.
            this.outbound_layer = outbound_layer;

            // The following 3 properties describe where
            // the input tensors come from: which layers,
            // and for each layer, which node and which
            // tensor output of each node.

            // List of layer instances.
            this.inbound_layers = inbound_layers;
            // List of integers, 1:1 mapping with inbound_layers.
            this.node_indices = node_indices;
            // List of integers, 1:1 mapping with inbound_layers.
            this.tensor_indices = tensor_indices;

            // Following 2 properties:
            // tensor inputs and outputs of outbound_layer.

            // List of tensors. 1:1 mapping with inbound_layers.
            this.input_tensors = input_tensors;
            // List of tensors, created by outbound_layer.call().
            this.output_tensors = output_tensors;

            // Following 2 properties: input and output masks.
            // List of tensors, 1:1 mapping with input_tensor.
            this.input_masks = input_masks;
            // List of tensors, created by outbound_layer.compute_mask().
            this.output_masks = output_masks;

            // Following 2 properties: input and output shapes.

            // List of shape tuples, shapes of input_tensors.
            this.input_shapes = input_shapes;
            // List of shape tuples, shapes of output_tensors.
            this.output_shapes = output_shapes;

            // Optional keyword arguments to layer's `call`.
            this.arguments = arguments;

            // Add nodes to all layers involved.
            foreach (Layer layer in inbound_layers)
            {
                if (layer != null)
                    layer.outbound_nodes.Add(this);
            }
            outbound_layer.inbound_nodes.Add(this);
        }

        //def get_config(self):
        //    inbound_names = []
        //        for layer in this.inbound_layers:
        //        if layer:
        //            inbound_names.append(layer.name)
        //            else:
        //            inbound_names.append(None)
        //        return {
        //            'outbound_layer': this.outbound_layer.name if this.outbound_layer else None,
        //            'inbound_layers': inbound_names,
        //            'node_indices': this.node_indices,
        //            'tensor_indices': this.tensor_indices}
        //    }

        public override string ToString()
        {
            string inputLayers = String.Join(", ", this.inbound_layers.Select(x => x.ToString()));
            string outputLayer = this.outbound_layer.ToString();

            return $"{{ {inputLayers} }} => {outputLayer}";
        }
    }
}
