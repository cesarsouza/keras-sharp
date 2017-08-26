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
    using Accord.Math.Random;
    using KerasSharp.Engine.Topology;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using System.Threading.Tasks;
    using TensorFlow;
    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   Applies Dropout to the input.
    /// </summary>
    /// 
    /// <remarks>
    ///   Dropout consists in randomly setting a fraction `rate` of input units to 0 at each 
    ///   update during training time, which helps prevent overfitting.
    /// </remarks>
    /// 
    /// <seealso cref="KerasSharp.IActivationFunction" />
    /// 
    [DataContract]
    public class Dropout : Layer
    {
        private double rate;
        private int[] noise_shape;
        private int? seed;

        /// <summary>
        /// Initializes a new instance of the <see cref="Dropout"/> class.
        /// </summary>
        /// <param name="rate">A float between 0 and 1. Fraction of the input units to drop.</param>
        /// <param name="noise_shape">1D integer tensor representing the shape of the binary dropout mask that will 
        ///   be multiplied with the input. For instance, if your inputs have shape <c>(batch_size, timesteps, features)</c>
        ///   and you want the dropout mask to be the same for all timesteps, you can use 
        ///   <c>noise_shape= (batch_size, 1, features)</c>.</param>
        /// <param name="seed">The integer to use as random seed.</param>
        /// 
        public Dropout(double rate, int[] noise_shape = null, int? seed = null)
        {
            this.rate = Math.Min(1.0, Math.Max(0.0, rate));
            this.noise_shape = noise_shape;
            this.seed = seed;
            this.supports_masking = true;
        }

        private int[] _get_noise_shape(Tensor x)
        {
            return this.noise_shape;
        }

        protected override Tensor InnerCall(Tensor inputs, Tensor mask, bool? training = null)
        {
            if (0.0 < this.rate && this.rate < 1.0)
            {
                var noise_shape = this._get_noise_shape(inputs);
                Func<Tensor> dropped_inputs = () => K.dropout(inputs, this.rate, noise_shape, seed: this.seed);
                return K.in_train_phase(dropped_inputs, () => inputs, training: training);
            }

            return inputs;
        }
    }
}
