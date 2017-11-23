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
    using static Python;
    using Accord.Math;

    using static KerasSharp.Backends.Current;
    public class conv_utils
    {
        /// <summary>
        ///   Transforms a single int or iterable of ints into an int tuple.
        /// </summary>
        /// <param name="value">The value to validate and convert. Could an int, or any iterable of ints.</param>
        /// <param name="n">The size of the tuple to be returned.</param>
        /// <param name="name">The name of the argument being validated, e.g. "strides" or "kernel_size".This is only used to format error messages.</param>
        /// <returns>System.Object.</returns>
        internal int[] normalize_tuple(int value, int n, string name)
        {
            return Vector.Create<int>(size: n, value: value);
        }

        /// <summary>
        ///   Transforms a single int or iterable of ints into an int tuple.
        /// </summary>
        /// <param name="value">The value to validate and convert. Could an int, or any iterable of ints.</param>
        /// <param name="n">The size of the tuple to be returned.</param>
        /// <param name="name">The name of the argument being validated, e.g. "strides" or "kernel_size".This is only used to format error messages.</param>
        /// <returns>System.Object.</returns>
        internal int[] normalize_tuple(int[] value_tuple, int n, string name)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/utils/conv_utils.py#L23

            if (len(value_tuple) != n)
                throw new Exception($"The {name} argument must be a tuple of {n} integers. Received: {value_tuple}");

            return value_tuple;
        }

        internal object normalize_data_format(DataFormatType? value)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/utils/conv_utils.py#L46

            if (value == null)
                value = K.image_data_format();

            return value;
        }

        /// <summary>
        ///   Determines output length of a convolution given input length.
        /// </summary>
        /// 
        public static int? conv_output_length(int? input_length, int filter_size, PaddingType padding, int stride, int dilation = 1)
        {
            if (input_length == null)
                return null;
            int dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1);
            int output_length = 0;
            if (padding == PaddingType.Same)
                output_length = input_length.Value;
            else if (padding == PaddingType.Valid)
                output_length = input_length.Value - dilated_filter_size + 1;
            else if (padding == PaddingType.Causal)
                output_length = input_length.Value;
            else if (padding == PaddingType.Full)
                output_length = input_length.Value + dilated_filter_size - 1;
            else
                throw new Exception();
            return (output_length + stride - 1); // stride
        }
    }
}
