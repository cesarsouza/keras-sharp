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
using Accord;
using System.IO;

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
        private List<TFOperation> updates_op;

        public TFFunction(TensorFlowBackend k, List<Tensor> inputs, List<Tensor> outputs, List<List<Tensor>> updates, string name)
        {
            this.K = k;
            this.tf = k.tf;

            if (updates == null)
                updates = new List<List<Tensor>>();
            this.inputs = inputs;
            this.outputs = outputs;
            {
                var updates_ops = new List<TFOperation>();
                foreach (List<Tensor> update in updates)
                {
                    if (update.Count == 2)
                    {
                        var p = K.In(update[0]);
                        var new_p = K.In(update[1]);
                        updates_ops.Add(tf.Assign(p, new_p).Operation);
                    }
                    else if (update.Count == 1)
                    {
                        // assumed already an op
                        updates_ops.Add(K.In(update[0]).output.Operation);
                    }
                    else
                    {
                        throw new NotSupportedException();
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

            var init = tf.GetGlobalVariablesInitializer();
            if (init.Length > 0)
            {
                Console.WriteLine("Initializing variables:");
                foreach (var op in init)
                {
                    Console.WriteLine(" - " + op.Name);
                    session.Run(new TFOutput[0], new TFTensor[0], new TFOutput[0], new[] { op });
                }

                Console.WriteLine("Operations:");
                foreach (var op in tf.GetEnumerator())
                    Console.WriteLine(" - " + op.Name);
                Console.WriteLine();
            }

            //Console.WriteLine("Before:");
            //PrintVariables(feed_dict, session);
            // Console.ReadKey();

            var runner = session.GetRunner();

            foreach (var o in this.outputs)
                runner.Fetch(K.In(o).output);

            foreach (var op in this.updates_op)
                runner.AddTarget(op);

            foreach (KeyValuePair<Tensor, Array> pair in feed_dict)
            {
                TensorFlowTensor t = K.In(pair.Key);
                runner.AddInput(t.output, pair.Value);
            }



            var updated = runner.Run();

            //Console.WriteLine();

            //foreach (var v in updated)
            //{
            //    object obj = v.GetValue();
            //    if (obj is float[,])
            //        Console.WriteLine((obj as float[,]).ToCSharp());
            //    else if (obj is float[])
            //        Console.WriteLine((obj as float[]).ToCSharp());
            //    else
            //        Console.WriteLine(obj);
            //}

            //Console.WriteLine();
            //Console.WriteLine();

            //Console.WriteLine("After:");
            //PrintVariables(feed_dict, session);

            return updated.Get(0, this.outputs.Count).Select(t => K.Out(t)).ToList();

            // Console.ReadKey();
        }

        private void PrintVariables(Dictionary<Tensor, Array> feed_dict, TFSession session)
        {
            string[] ops =
            {
                //"SGD/grad/dense_1/dense_1/kernel/var",
                //"SGD/grad/dense_2/dense_2/kernel/var",
                //"SGD/grad/dense_2/dense_2/bias/var",
                //"loss/dense_1_loss/y_true",
                //"loss/dense_1_loss/y_pred",
                //"loss/dense_1_loss/weights",
                //"iterations/var",
                //"lr/var",
                //"lr_t",
                //"p_t",
                //"metrics/binary_accuracy/Round0",
                //"metrics/binary_accuracy/Cast0",
                //"metrics/binary_accuracy/Mean0",
                //"metrics/binary_accuracy/Equal0",
                //"metrics/binary_accuracy/value",
                //"metrics/score_array/mean"
                //"beta_1/var",
                //"beta_2/var",
                //"decay/var",
                //"adam/grad/dense_1/dense_1/kernel/var",
                //"dense_1/variance_scaling/1/scaled",
                //"dense_1/dense_1/kernel/var",
                //"dense_1/call/dot",
                //"dense_1/call/Sigmoid0",
            };

            foreach (var op in ops)
            {
                try
                {
                    var debugRunner = session.GetRunner();
                    foreach (KeyValuePair<Tensor, Array> pair in feed_dict)
                    {
                        TensorFlowTensor t = K.In(pair.Key);
                        debugRunner.AddInput(t.output, pair.Value);
                    }

                    Console.WriteLine(op);
                    debugRunner.Fetch(op);

                    var v = debugRunner.Run();

                    object obj = v[0].GetValue();

                    if (obj is float[,])
                        Console.WriteLine((obj as float[,]).ToCSharp());
                    else if (obj is float[])
                        Console.WriteLine((obj as float[]).ToCSharp());
                    else if (obj is bool[,])
                        Console.WriteLine((obj as bool[,]).ToCSharp());
                    else if (obj is bool[])
                        Console.WriteLine((obj as bool[]).ToCSharp());
                    else if (obj is sbyte[,])
                        Console.WriteLine((obj as sbyte[,]).ToCSharp());
                    else if (obj is sbyte[])
                        Console.WriteLine((obj as sbyte[]).ToCSharp());
                    else
                        Console.WriteLine(obj);
                }
                catch
                {

                }
            }
        }
    }
}