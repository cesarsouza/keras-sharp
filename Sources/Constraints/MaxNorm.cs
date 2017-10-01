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

namespace KerasSharp.Constraints
{
    using KerasSharp.Engine.Topology;
    using System.Runtime.Serialization;
    
    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   MaxNorm weight constraint.
    /// </summary>
    /// 
    /// <remarks>
    ///   Constrains the weights incident to each hidden unit
    ///   to have a norm less than or equal to a desired value.
    /// </remarks>
    /// 
    [DataContract]
    public class MaxNorm : IWeightConstraint
    {
        private int max_value;
        private int axis;

        /// <summary>
        /// Initializes a new instance of the <see cref="MaxNorm"/> class.
        /// </summary>
        /// <param name="max_value">The maximum value.</param>
        /// <param name="axis">The axis along which to calculate weight norms. For instance, in a <see cref="Dense"/> layer 
        ///   the weight matrix has shape <c>(input_dim, output_dim)</c>, set <paramref="axis"/> to <c>0</c> to constrain 
        ///   each weight vector of length <c>(input_dim,)</c>. In a <see cref="Convolution2D"/> layer with <c>data_format="channels_last"</c>,
        ///   the weight tensor has shape <c>(rows, cols, input_depth, output_depth)</c>, set <paramref="axis"/> to <c>[0, 1, 2]</c> to 
        ///   constrain the weights of each filter tensor of size <c>(rows, cols, input_depth)</c>.</param>
        ///   
        public MaxNorm(int max_value = 2, int axis = 0)
        {
            this.max_value = max_value;
            this.axis = axis;
        }

        /// <summary>
        /// Wires the constraint to the graph.
        /// </summary>
        /// 
        /// <param name="w">The weights tensor.</param>
        /// 
        /// <returns>The output tensor with the constraint applied.</returns>
        /// 
        public Tensor Call(Tensor w)
        {
            Tensor norms = K.sqrt(K.sum(K.square(w), axis: this.axis, keepdims: true));
            Tensor desired = K.clip(norms, 0, this.max_value);
            w = K.mul(w, K.div(desired, K.add(K.epsilon(), norms)));
            return w;
        }
    }
}