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

    [DataContract]
    public class Adam : OptimizerBase, IOptimizer
    {
        private Tensor iterations;
        private Tensor lr;
        private Tensor beta_1;
        private Tensor beta_2;
        private Tensor decay;
        private double initial_decay;
        private double epsilon;

        public Adam(double lr = 0.001, double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double decay = 0.0)
        {
            this.iterations = K.variable(0, name: "iterations");
            this.lr = K.variable(lr, name: "lr");
            this.beta_1 = K.variable(beta_1, name: "beta_1");
            this.beta_2 = K.variable(beta_2, name: "beta_2");
            this.epsilon = epsilon;
            this.decay = K.variable(decay, name: "decay");
            this.initial_decay = decay;
        }

        public List<List<Tensor>> get_updates(List<Tensor> param, Dictionary<Tensor, IWeightConstraint> constraints, Tensor loss)
        {
            var grads = this.get_gradients(loss, param);
            this.updates = new List<List<Tensor>> { new List<Tensor> { K.update_add(this.iterations, 1) } };

            Tensor lr = this.lr;
            if (this.initial_decay > 0)
                lr *= (1.0 / (1.0 + this.decay * this.iterations));

            Tensor t = this.iterations + 1;
            Tensor lr_t = lr * (K.sqrt(1.0 - K.pow(this.beta_2, t)) /
                         (1.0 - K.pow(this.beta_1, t)));

            var shapes = param.Select(p => K.get_variable_shape(p));
            var ms = shapes.Select(shape => K.zeros(shape)).ToArray();
            var vs = shapes.Select(shape => K.zeros(shape)).ToArray();
            this.weights = new[] { this.iterations }.Concat(ms).Concat(vs).ToList();

            for (int i = 0; i < param.Count; i++)
            {
                var p = param[i];
                var g = grads[i];
                var m = ms[i];
                var v = vs[i];
                var m_t = (this.beta_1 * m) + (1.0 - this.beta_1) * g;
                var v_t = (this.beta_2 * v) + (1.0 - this.beta_2) * K.square(g);
                var p_t = p - lr_t * m_t / (K.sqrt(v_t) + this.epsilon);

                this.updates.Add(new List<Tensor> { K.update(m, m_t) });
                this.updates.Add(new List<Tensor> { K.update(v, v_t) });

                var new_p = p_t;
                // apply constraints
                if (constraints.Keys.Contains(p))
                {
                    var c = constraints[p];
                    new_p = c.Call(new_p);
                    this.updates.Add(new List<Tensor> { K.update(p, new_p) });
                }
            }

            return this.updates;
        }
    }
}