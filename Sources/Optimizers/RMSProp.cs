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
    using KerasSharp.Models;
    using System;
    using System.Collections.Generic;
    using System.Runtime.Serialization;

    using static KerasSharp.Backends.Current;
    using KerasSharp.Engine.Topology;
    using System.Linq;

    /// <summary>
    ///   RMSProp optimizer.
    /// </summary>
    /// 
    /// <remarks>
    ///   It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
    /// </remarks>
    /// 
    [DataContract]
    public class RMSProp : OptimizerBase, IOptimizer
    {
        private Tensor decay;
        private double initial_decay;
        private Tensor iterations;
        private Tensor lr;
        private Tensor rho;
        private double epsilon;

        // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/optimizers.py#L190

        public RMSProp()
            : this(lr: 0.001, rho: 0.9, epsilon: 1e-8, decay: 0.0)
        {

        }

        public RMSProp(double lr, double rho = 0.9, double epsilon = 1e-8, double decay = 0.0)
        {
            this.lr = K.variable(lr, name: "lr");
            this.rho = K.variable(rho, name: "rho");
            this.epsilon = epsilon;
            this.decay = K.variable(decay, name: "decay");
            this.initial_decay = decay;
            this.iterations = K.variable(0.0, name: "iterations");
        }

        public List<List<Tensor>> get_updates(List<Tensor> parameters, Dictionary<Tensor, IWeightConstraint> constraints, Tensor loss)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/optimizers.py#L221

            List<Tensor> grads = this.get_gradients(loss, parameters);
            List<int?[]> shapes = parameters.Select(p => K.get_variable_shape(p)).ToList();
            List<Tensor> accumulators = shapes.Select(shape => K.zeros(shape)).ToList();
            this.weights = accumulators;
            this.updates = new List<List<Tensor>>();

            Tensor lr = this.lr;
            if (this.initial_decay > 0)
            {
                lr = lr * (1.0 / (1.0 + this.decay * this.iterations));
                this.updates.Add(K.update_add(this.iterations, 1));
            }

            for (int i = 0; i < parameters.Count; i++)
            {
                Tensor p = parameters[i];
                Tensor g = grads[i];
                Tensor a = accumulators[i];

                // update accumulator
                Tensor new_a = this.rho * a + (1.0 - this.rho) * K.square(g);
                this.updates.Add(K.update(a, new_a));
                Tensor new_p = p - lr * g / (K.sqrt(new_a) + this.epsilon);

                // apply constraints
                if (constraints.ContainsKey(p))
                {
                    IWeightConstraint c = constraints[p];
                    new_p = c.Call(new_p);
                }

                this.updates.Add(K.update(p, new_p));
            }

            return this.updates;
        }
    }
}