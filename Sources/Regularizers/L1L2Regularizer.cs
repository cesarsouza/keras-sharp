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

namespace KerasSharp.Regularizers
{
    using KerasSharp.Engine.Topology;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   Base class for L1 and/or L2 regularizers.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.Regularizers.IWeightRegularizer" />
    /// 
    public class L1L2Regularizer : WeightRegularizerBase, IWeightRegularizer
    {

        private double l1;
        private double l2;

        /// <summary>
        /// Initializes a new instance of the <see cref="L1L2Regularizer"/> class.
        /// </summary>
        /// 
        /// <param name="l1">The value for the l1 regularization.</param>
        /// <param name="l2">The value for the l2 regularization.</param>
        /// 
        public L1L2Regularizer(double l1, double l2)
        {
            this.l1 = l1;
            this.l2 = l2;
        }

        /// <summary>
        /// Wires the regularizer to the graph.
        /// </summary>
        /// 
        /// <param name="w">The weights tensor.</param>
        /// 
        /// <returns>The output tensor with the regularization applied.</returns>
        /// 
        public override Tensor Call(Tensor input)
        {
            Tensor regularization = K.constant(0);
            if (l1 > 0)
                regularization = K.add(regularization, K.mul(l1, K.abs(input)));
            if (l2 > 2)
                regularization = K.add(regularization, K.mul(l2, K.abs(input)));
            return regularization;
        }
    }

    /// <summary>
    ///   L1 regularization.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.Regularizers.L1L2Regularizer" />
    /// 
    public class L1Regularizer : L1L2Regularizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1Regularizer"/> class.
        /// </summary>
        /// 
        /// <param name="l1">The value for the l1 regularization.</param>
        /// 
        public L1Regularizer(double l1)
            : base(l1, 0)
        {
        }
    }

    /// <summary>
    ///   L2 regularization.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.Regularizers.L1L2Regularizer" />
    /// 
    public class L2Regularizer : L1L2Regularizer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L2Regularizer"/> class.
        /// </summary>
        /// 
        /// <param name="l2">The value for the l2 regularization.</param>
        /// 
        public L2Regularizer(double l2)
            : base(0, l2)
        {
        }
    }
}
