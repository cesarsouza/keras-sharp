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

namespace KerasSharp.Initializers
{
    using Accord.Math;
    using KerasSharp.Engine.Topology;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;

    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   Initializer that generates tensors initialized to 1.
    /// </summary>
    /// 
    [DataContract]
    public class Ones : IWeightInitializer
    {

        /// <summary>
        /// Creates a <see cref="TFTensor" /> with the desired initial weights.
        /// </summary>
        /// 
        /// <param name="shape">The shape of the tensor to be generated.</param>
        /// <param name="dtype">The <see cref="TFDataType">data type</see> of the tensor to be generated.</param>
        /// <returns>A <see cref="TFTensor" /> initialized of dimensions <paramref name="shape" />
        /// and element data type <paramref name="dtype" /> that has been initialized using this
        /// strategy.</returns>
        /// 
        public Tensor Call(int?[] shape, DataType? dtype = null)
        {
            return K.constant(1, shape: shape, dtype: dtype);
        }
    }
}
