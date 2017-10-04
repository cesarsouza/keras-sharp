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

namespace KerasSharp.Optimizers
{
    using KerasSharp.Constraints;
    using KerasSharp.Engine.Topology;
    using KerasSharp.Losses;
    using KerasSharp.Models;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;

    using static KerasSharp.Backends.Current;


    /// <summary>
    ///   Stochastic gradient descent optimizer.
    /// </summary>
    /// 
    /// <remarks>
    ///  Includes support for momentum, learning rate decay, and Nesterov momentum.
    /// </remarks>
    /// 
    /// <seealso cref="KerasSharp.Models.IOptimizer" />
    /// 
    [DataContract]
    public class SGD : OptimizerBase, IOptimizer
    {
        private Tensor iterations;
        private Tensor lr;
        private Tensor momentum;
        private Tensor decay;
        private double initial_decay;
        private bool nesterov;

        /// <summary>
        /// Initializes a new instance of the <see cref="SGD" /> class.
        /// </summary>
        /// 
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="momentum">float >= 0. Parameter updates momentum.</param>
        /// <param name="decay">float >= 0. Learning rate decay over each update.</param>
        /// <param name="nesterov">Whether to apply Nesterov momentum.</param>
        /// 
        public SGD(double lr = 0.01, double momentum = 0.0, double decay = 0.0, bool nesterov = false)
            : base()
        {
            this.iterations = K.variable(0.0, name: "iterations");
            this.lr = K.variable(lr, name: "lr");
            this.momentum = K.variable(momentum, name: "momentum");
            this.decay = K.variable(decay, name: "decay");
            this.initial_decay = decay;
            this.nesterov = nesterov;
        }

        public List<List<Tensor>> get_updates(List<Tensor> param, Dictionary<Tensor, IWeightConstraint> constraints, Tensor loss)
        {
            var grads = this.get_gradients(loss, param);
            this.updates = new List<List<Tensor>>();

            var lr = this.lr;
            if (this.initial_decay > 0)
            {
                lr = K.mul(lr, K.div(1.0, K.sum(1.0, K.mul(this.decay, this.iterations))));
                this.updates.Add(K.update_add(this.iterations, 1));
            }

            // momentum
            List<int?[]> shapes = param.Select(p => K.get_variable_shape(p)).ToList();
            List<Tensor> moments = shapes.Select(s => K.zeros(s)).ToList();

            this.weights = new[] { this.iterations }.Concat(moments).ToList();

            for (int i = 0; i < param.Count; i++)
            {
                Tensor p = param[i];
                Tensor g = grads[i];
                Tensor m = moments[i];
                Tensor v = K.subtract(K.mul(this.momentum, m), K.mul(lr, g));  // velocity

                this.updates.Add(K.update(m, v));

                Tensor new_p;
                if (this.nesterov)
                    new_p = K.add(p, K.subtract(K.mul(this.momentum, v), K.mul(lr, g)));
                else
                    new_p = K.add(p, v);

                // apply constraints

                if (constraints.ContainsKey(p))
                {
                    var c = constraints[p];
                    new_p = c.Call(new_p);
                }


                updates.Add(K.update(p, new_p));
            }

            return this.updates;
        }

    }
}