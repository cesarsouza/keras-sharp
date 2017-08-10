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

namespace KerasSharp.Layers
{
    using Accord.Math;
    using KerasSharp.Engine.Topology;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;

    using static KerasSharp.Backends.Current;


    /// <summary>
    /// Layer to be used as an entry point into a graph.
    /// </summary>
    /// 
    /// <remarks>
    ///  It can either wrap an existing tensor (pass an `input_tensor` argument)
    ///  or create its a placeholder tensor(pass arguments `input_shape`
    ///  or `batch_input_shape` as well as `dtype`).
    /// </remarks>
    /// 
    /// <seealso cref="KerasSharp.Engine.Topology.Layer" />
    /// 
    public class InputLayer : Layer
    {
        private bool sparse;
        private Tensor input_tensor;


        /// <summary>
        /// Initializes a new instance of the <see cref="InputLayer"/> class.
        /// </summary>
        /// 
        /// <param name="input_shape">Shape vector, not including the batch axis.</param>
        /// <param name="batch_size">Optional input batch size (integer or null).</param>
        /// <param name="batch_input_shape">Shape vector, including the batch axis.</param>
        /// <param name="name">The name of the layer.</param>
        /// <param name="dtype">The datatype of the input.</param>
        /// <param name="sparse">Whether the placeholder created is meant to be sparse.</param>
        /// <param name="input_tensor">The optional tensor to use as layer input.</param>
        /// 
        public InputLayer(int?[] input_shape = null, int? batch_size = null, int?[] batch_input_shape = null, string name = null,
            TFDataType? dtype = null, bool sparse = false, Tensor input_tensor = null)
            : base(dtype: GetType(dtype, input_tensor), name: GetName(name))
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/engine/topology.py#L1291

            this.batch_input_shape = batch_input_shape;
            this.name = name;
            this.sparse = sparse;
            this.input_tensor = input_tensor;

            this.trainable = false;
            this.built = true;
            this.sparse = sparse;

            if (input_shape != null && batch_input_shape != null)
                throw new Exception("Only provide the input_shape OR batch_input_shape argument to  InputLayer, not both at the same time.");

            if (input_tensor != null && batch_input_shape == null)
            {
                // If input_tensor is set, and batch_input_shape is not set:
                // Attempt automatic input shape inference.
                try
                {
                    batch_input_shape = K.int_shape(input_tensor);
                }
                catch
                {
                    if (input_shape == null && batch_input_shape == null)
                        throw new Exception("InputLayer was provided an input_tensor argument, but its input shape " +
                            "cannot be automatically inferred. You should pass an input_shape or batch_input_shape argument.");
                }

                if (batch_input_shape == null)
                {
                    if (input_shape == null)
                        throw new Exception("An Input layer should be passed either a `batch_input_shape` or an `input_shape`.");
                }
                else
                {
                    batch_input_shape = batch_size.Concatenate(input_shape);
                }
            }


            this.batch_input_shape = batch_input_shape;
            this.dtype = dtype.Value;


            if (input_tensor == null)
            {
                this.is_placeholder = true;
                this.input_tensor = K.placeholder(shape: batch_input_shape,
                                        dtype: dtype,
                                        sparse: this.sparse,
                                        name: this.name);
            }
            else
            {
                this.is_placeholder = false;
                this.input_tensor._keras_shape = batch_input_shape;
            }

            // Create an input node to add to this.outbound_node
            // and set output_tensors" _keras_history.
            this.input_tensor._uses_learning_phase = false;
            this.input_tensor._keras_history = (this, 0, 0);

            var node = new Node(this,
                inbound_layers: new List<Layer>(),
                node_indices: new List<int?>(),
                tensor_indices: new List<int?>(),
                input_tensors: new List<Tensor> { this.input_tensor },
                output_tensors: new List<Tensor> { this.input_tensor },
                input_masks: new List<Tensor> { null },
                output_masks: new List<Tensor> { null },
                input_shapes: new List<int?[]> { batch_input_shape },
                output_shapes: new List<int?[]> { batch_input_shape });
        }

        private static TFDataType? GetType(TFDataType? dtype, TFTensor input_tensor)
        {
            throw new NotImplementedException();
        }

        private static TFDataType GetType(TFDataType? dtype, Tensor input_tensor)
        {
            if (dtype == null)
            {
                if (input_tensor == null)
                    dtype = K.floatx();
                else
                    dtype = K.dtype(input_tensor);
            }

            return dtype.Value;
        }

        private static string GetName(string name)
        {
            string prefix = "";
            if (name == null)
                prefix = "input";
            name = prefix + "_" + K.get_uid(prefix);
            return name;
        }

    }
}
