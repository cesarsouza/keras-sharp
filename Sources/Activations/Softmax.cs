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

namespace KerasSharp.Activations
{
    using static KerasSharp.Backends.Current;

    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;
    using System.Runtime.Serialization;
    using KerasSharp.Engine.Topology;

    /// <summary>
    ///   Softmax activation function.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.IActivationFunction" />
    /// 
    [DataContract]
    public class Softmax : ActivationFunctionBase, IActivationFunction
    {
        private int axis;

        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax"/> class.
        /// </summary>
        /// 
        public Softmax()
            : this(-1)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax"/> class.
        /// </summary>
        /// 
        /// <param name="axis">The axis along which the softmax normalization is applied.</param>
        /// 
        public Softmax(int axis)
        {
            this.axis = axis;
        }

        /// <summary>
        /// Wires the activation function to the graph.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The output tensor with the activation function applied.</returns>
        /// <exception cref="System.ArgumentException">Cannot apply softmax to a tensor that is 1D</exception>
        public override Tensor Call(Tensor x, Tensor mask = null)
        {
            int? ndim = K.ndim(x);
            if (ndim == 2)
                return K.softmax(x);

            if (ndim > 2)
            {
                Tensor e = K.exp(x - K.max(x, axis: axis, keepdims: true));
                Tensor s = K.sum(e, axis: axis, keepdims: true);
                return K.div(e, s);
            }

            throw new ArgumentException("Cannot apply softmax to a tensor that is 1D");
        }
    }
}
