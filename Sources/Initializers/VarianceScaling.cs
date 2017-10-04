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
    ///   Initializer capable of adapting its scale to the shape of weights.
    /// </summary>
    /// <remarks>
    ///  With `distribution="normal"`, samples are drawn from a truncated normal distribution 
    ///  centered on zero, with `stddev = sqrt(scale / n)` where n is:
    ///  <list type="bullet">  
    ///     <listheader>  
    ///         <term></term><description>number of input units in the weight tensor, if mode = "fan_in"</description>  
    ///     </listheader>  
    ///     <item>  
    ///         <term></term><description>number of input units in the weight tensor, if mode = "fan_in"</description>  
    ///     </item>  
    ///     <item>  
    ///         <term>term</term><description>number of output units, if mode = "fan_out"</description>  
    ///     </item>  
    ///     <item>  
    ///         <term>term</term><description>average of the numbers of input and output units, if mode = "fan_avg"</description>  
    ///     </item>  
    ///     <item>  
    ///         <term>term</term><description>With `distribution="uniform"`, samples are drawn from a uniform distribution
    ///           within[-limit, limit], with `limit = sqrt(3 * scale / n)`.</description>  
    ///     </item>  
    /// </list>  
    /// </remarks>
    /// 
    [DataContract]
    public class VarianceScaling : IWeightInitializer
    {
        private double scale;
        private string mode;
        private string distribution;
        private int? seed;

        /// <summary>
        /// Initializes a new instance of the <see cref="VarianceScaling" /> class.
        /// </summary>
        /// 
        /// <param name="scale">The Scaling factor (positive float).</param>
        /// <param name="mode">The mode, one of "fan_in", "fan_out", "fan_avg".</param>
        /// <param name="distribution">The random distribution to use. One of "normal", "uniform".</param>
        /// <param name="seed">The seed used to seed the random generator.</param>
        /// 
        public VarianceScaling(double scale = 1.0, string mode = "fan_in", string distribution = "normal", int? seed = null)
        {
            if (scale <= 0.0)
                throw new ArgumentOutOfRangeException("scale must be a positive float. Got: " + scale);

            mode = mode.ToLowerInvariant();

            if (!new[] { "fan_in", "fan_out", "fan_avg" }.Contains(mode))
                throw new ArgumentException("Invalid `mode` argument: expected on of {\"fan_in\", \"fan_out\", \"fan_avg\"}  but got " + mode);

            distribution = distribution.ToLowerInvariant();

            if (!new[] { "normal", "uniform" }.Contains(distribution))
                throw new ArgumentException("Invalid `distribution` argument: expected one of {\"normal\", \"uniform\"} but got " + distribution);

            this.scale = scale;
            this.mode = mode;
            this.distribution = distribution;
            this.seed = seed;
        }

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
            var (fan_in, fan_out) = _compute_fans(shape.Select(x => x.Value).ToArray());

            Double scale = this.scale;
            if (this.mode == "fan_in")
                scale /= Math.Max(1.0, fan_in);
            else if (this.mode == "fan_out")
                scale /= Math.Max(1.0, fan_out);
            else
                scale /= Math.Max(1.0, (fan_in + fan_out) / 2.0);

            if (this.distribution == "normal")
            {
                var stddev = Math.Sqrt(scale);
                return K.truncated_normal(shape, 0.0, stddev, dtype: dtype, seed: this.seed);
            }
            else
            {
                var limit = Math.Sqrt(3.0 * scale);
                return K.random_uniform(shape, -limit, limit, dtype: dtype, seed: this.seed);
            }
        }


        /// <summary>
        ///   Computes the number of input and output units for a weight shape.
        /// </summary>
        /// 
        /// <param name="shape">The shape array.</param>
        /// <param name="data_format">The image data format to use for convolution kernels. Note that all kernels 
        ///   in Keras are standardized on the <c>channels_last</c> ordering(even when inputs are set to
        ///   <c>channels_first</c>).</param>
        /// 
        private (double fan_int, double fan_out) _compute_fans(int[] shape, string data_format = "channels_last")
        {
            double fan_in, fan_out;
            if (shape.Length == 2)
            {
                fan_in = shape[0];
                fan_out = shape[1];
            }
            else if (new[] { 3, 4, 5 }.Contains(shape.Length))
            {
                // Assuming convolution kernels (1D, 2D or 3D).
                // TH kernel shape: (depth, input_depth, ...)
                // TF kernel shape: (..., input_depth, depth)
                if (data_format == "channels_first")
                {
                    int receptive_field_size = shape.Get(2, 0).Product();
                    fan_in = shape[1] * receptive_field_size;
                    fan_out = shape[0] * receptive_field_size;
                }
                else if (data_format == "channels_last")
                {
                    int receptive_field_size = shape.Get(0, 2).Product();
                    fan_in = shape.Get(-2) * receptive_field_size;
                    fan_out = shape.Get(-1) * receptive_field_size;
                }
                else
                {
                    throw new ArgumentOutOfRangeException("Invalid data_format: " + data_format);
                }
            }
            else
            {
                // No specific assumptions.
                fan_in = Math.Sqrt(shape.Product());
                fan_out = Math.Sqrt(shape.Product());
            }

            return (fan_in, fan_out);
        }
    }
}
