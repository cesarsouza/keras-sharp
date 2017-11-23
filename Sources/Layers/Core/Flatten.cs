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

namespace KerasSharp
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    using System.Runtime.Serialization;
    using KerasSharp.Constraints;
    using KerasSharp.Regularizers;
    using KerasSharp.Initializers;
    using Accord.Math;
    using KerasSharp.Engine.Topology;

    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   Flattens the input. Does not affect the batch size.
    /// </summary>
    /// <seealso cref="KerasSharp.Engine.Topology.Layer" />
    [DataContract]
    public class Flatten : Layer
    {
        protected override Tensor InnerCall(Tensor inputs, Tensor mask = null, bool? training = null)
        {
            return K.batch_flatten(inputs);
        }

        public override List<int?[]> compute_output_shape(List<int?[]> input_shapes)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/core.py#L473
            if (input_shapes.Count > 0)
                throw new Exception();

            var input_shape = input_shapes[0];

            if (!input_shape.Get(1, 0).All(x => x > 0))
            {
                throw new Exception($"The shape of the input to 'Flatten' is not fully defined  (got {input_shape.Get(1, 0)}). " +
                    $"Make sure to pass a complete {input_shape} or {batch_input_shape} argument to the first layer in your model.");
            }

            return new List<int?[]> { new int?[] { input_shape[0], Matrix.Product(input_shape.Select(x=>x.Value).ToArray().Get(1, 0)) } };
        }
    }
}
