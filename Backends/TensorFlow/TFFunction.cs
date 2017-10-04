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

using System;
using System.Collections.Generic;
using KerasSharp.Engine.Topology;
using KerasSharp.Models;
using TensorFlow;
using System.Linq;
using Accord.Math;

namespace KerasSharp.Backends
{

    /// <summary>
    ///   Runs a computation graph.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.Models.Function" />
    /// 
    public class TFFunction : Function
    {
        private TensorFlowBackend K;
        private TFGraph tf;
        private List<Tensor> inputs;
        private List<Tensor> outputs;
        private string name;
        private List<TFOutput> updates_op;

        public TFFunction(TensorFlowBackend k, List<Tensor> inputs, List<Tensor> outputs, List<List<Tensor>> updates, string name)
        {
            this.K = k;
            this.tf = k.tf;

            if (updates == null)
                updates = new List<List<Tensor>>();
            this.inputs = inputs;
            this.outputs = outputs;
            {
                var updates_ops = new List<TFOutput>();
                foreach (List<Tensor> update in updates)
                {
                    if (update.Count == 2)
                    {
                        var p = K.In(update[0]);
                        var new_p = K.In(update[1]);
                        updates_ops.Add(tf.Assign(p, new_p));
                    }
                    else
                    {
                        // assumed already an op
                        updates_ops.Add(K.In(update[0]));
                    }
                }

                //this.updates_op = tf.group(updates_ops);
                this.updates_op = updates_ops;
            }

            this.name = name;
            //this.session_kwargs = session_kwargs;
        }

        public override List<Tensor> Call(List<Array> inputs)
        {
            var feed_dict = new Dictionary<Tensor, Array>();
            foreach (var (tensor, value) in Enumerable.Zip(this.inputs, inputs, (a, b) => (a, b)))
            {
                // if (is_sparse(tensor))
                // {
                //     sparse_coo = value.tocoo()
                //     indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                //                               np.expand_dims(sparse_coo.col, 1)), 1)
                //     value = (indices, sparse_coo.data, sparse_coo.shape)
                // }
                feed_dict[tensor] = value;
            }

            var session = K._SESSION;
            var outputs = new List<TFOutput>();
            foreach (var o in this.outputs)
                outputs.Add(K.In(o));
            foreach (TFOutput o in this.updates_op)
                outputs.Add(o);

            var _inputs = new List<TFOutput>();
            var _values = new List<TFTensor>();
            foreach (KeyValuePair<Tensor, Array> pair in feed_dict)
            {
                _inputs.Add(K.In(pair.Key));
                _values.Add(pair.Value);
            }

            var updated = session.Run(
                inputs: _inputs.ToArray(), 
                inputValues: _values.ToArray(),
                outputs: outputs.ToArray());

            return updated.Get(0, this.outputs.Count).Select(t => K.Out(t)).ToList();
        }
    }
}