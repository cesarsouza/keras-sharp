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
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;
    

    using static KerasSharp.Python;

    [DataContract]
    public class InputSpec
    {
        public DataType? dtype;
        public int[] shape;
        public int? ndim;
        public int? max_ndim;
        public int? min_ndim;
        public Dictionary<int, int> axes;

        /// <summary>
        ///   Specifies the ndim, dtype and shape of every input to a layer.
        /// </summary>
        /// 
        /// <remarks>
        ///   Every layer should expose (if appropriate) an `input_spec` attribute:
        ///   a list of instances of InputSpec(one per input tensor).
        ///   A null entry in a shape is compatible with any dimension,
        ///   a null shape is compatible with any shape.
        /// </remarks>
        /// 
        /// <param name="dtype">Expected datatype of the input.</param>
        /// <param name="shape">Shape tuple, expected shape of the input
        ///         (may include null for unchecked axes).</param>
        /// <param name="ndim">Integer, expected rank of the input.</param>
        /// <param name="max_ndim">The maximum rank of the input.</param>
        /// <param name="min_ndim">The minimum rank of the input.</param>
        /// <param name="axes">Dictionary mapping integer axes to
        ///    a specific dimension value.</param>
        public InputSpec(DataType? dtype = null, int[] shape = null, int? ndim = null,
                         int? max_ndim = null, int? min_ndim = null, Dictionary<int, int> axes = null)
        {
            this.dtype = dtype;
            this.shape = shape;
            if (shape != null)
                this.ndim = shape.Length;
            else
                this.ndim = ndim;
            this.max_ndim = max_ndim;
            this.min_ndim = min_ndim;
            this.axes = axes ?? new Dictionary<int, int>();
        }

        public override string ToString()
        {
            return $"dtype={dtype}, shape={str(shape)}, ndim={ndim}, max_ndim={max_ndim}, min_ndim={min_ndim}, axes={str(axes)}";
        }
    }
}
