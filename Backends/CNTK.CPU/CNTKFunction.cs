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
using CNTK;
using KerasSharp.Engine.Topology;
using KerasSharp.Models;
using static KerasSharp.Python;
using C = CNTK.CNTKLib;
using System.Linq;
using Accord.Math;

namespace KerasSharp.Backends
{
    internal class CNTKFunction : Models.Function
    {
        private List<List<Tensor>> updates;
        private List<Variable> placeholders;
        private Trainer trainer;
        private UnorderedMapVariableValuePtr trainer_output;
        private CNTK.Function unrelated_updates;
        private Variable[] metrics_outputs;
        private CNTK.Function metrics_func;
        private CNTK.Function loss;
        private CNTKBackend c;

        public CNTKFunction(CNTKBackend c, List<Variable> inputs, CNTK.Function[] outputs, List<List<Tensor>> updates, string name)
        {
            this.c = c;
            this.placeholders = inputs;
            this.trainer = null;
            this.unrelated_updates = null;
            this.updates = updates;
            if (updates.Count > 0)
            {
                if (len(outputs) <= 0)
                    throw new Exception();

                this.loss = outputs[0];
                // need group update by gradient place holder
                var u_ops = new List<CNTK.Function>();
                var unrelated_updates = new List<CNTK.Function>();
                foreach (List<Tensor> update in updates)
                {
                    CNTK.Function u;

                    if (update.Count == 1)
                    {
                        u = c.In(update[0]);
                    }
                    else if (update.Count == 2)
                    {
                        u = C.Assign(c.In(update[0]), c.In(update[1]));
                    }
                    else
                    {
                        throw new NotImplementedException();
                    }

                    if (u.Inputs.Count == 0)
                        u_ops.Add(u);
                    else
                        unrelated_updates.Add(u);
                }

                var update_func = C.Combine(new VariableVector(u_ops.Select(u => u.Output).ToArray()));

                CNTK.Function[] grads = update_func.Inputs.Where(x => x.Name == "keras_grad_placeholder").Select(x => x.ToFunction()).ToArray();
                //CNTK.Function[] grads = update_func.FindAllWithName("keras_grad_placeholder").ToArray();

                var u_list = new List<CNTK.Function>();
                var p_list = new List<CNTK.Parameter>();
                foreach (CNTK.Function g in grads)
                {
                    if (c.grad_parameter_dict.ContainsKey(g))
                    {
                        p_list.Add(c.grad_parameter_dict[g]);
                        u_list.Add(g);
                    }
                    else
                    {
                        throw new Exception($"CNTK backend: when constructing trainer, found gradient node {g} which is not related to any parameters in the model. Please double check how the gradient node is constructed.");
                    }
                }

                if (len(u_list) > 0)
                {
                    Learner learner = Learner.SGDLearner(p_list, new TrainingParameterScheduleDouble(0));

                    var criterion = (len(outputs) > 1) ?
                        C.Combine(new VariableVector(new[] { outputs[0], outputs[1] })) :
                        outputs[0];

                    this.trainer = Trainer.CreateTrainer(model: outputs[0], lossFunction: criterion, evaluationFunction: null, parameterLearners: new[] { learner });

                    this.trainer_output = new UnorderedMapVariableValuePtr();
                    foreach (CNTK.Function f in outputs)
                        this.trainer_output.Add(f, null);
                }
                else if (len(u_ops) > 0)
                {
                    unrelated_updates.AddRange(u_ops);
                }

                if (len(unrelated_updates) > 0)
                    this.unrelated_updates = C.Combine(new VariableVector(unrelated_updates.Select(_ => _.Output).ToArray()));
            }

            if (this.trainer == null)
            {
                this.metrics_outputs = outputs.Select(f => f.Output).ToArray();

                this.metrics_func = C.Combine(new VariableVector(this.metrics_outputs));
                // cntk only could handle loss and 1 metric in trainer, for metrics more
                // than 2, need manual eval
            }
            else if (len(outputs) > 2)
            {
                this.metrics_outputs = Matrix.Get(outputs, 2, 0).Select(f => f.Output).ToArray();

                this.metrics_func = C.Combine(new VariableVector(this.metrics_outputs));
            }
            else
            {
                this.metrics_func = null;
            }
        }

        public override List<Tensor> Call(List<Array> inputs)
        {
            var feed_dict = new Dictionary<Variable, Array>();
            foreach (var (tensor, value) in Enumerable.Zip(this.placeholders, inputs, (a, b) => (a, b)))
            {
                Type t = value.GetInnerMostType();
                Array v = value;

                // cntk only support calculate on float, do auto cast here
                if (t != typeof(float) && t != typeof(double))
                    v = MatrixEx.Convert<double>(value);
                feed_dict[tensor] = v;
            }

            var updated = new List<Tensor>();
            if (this.trainer != null)
            {
                var input_dict = new UnorderedMapVariableValuePtr();
                foreach (Variable argument in this.loss.Arguments)
                {
                    if (feed_dict.ContainsKey(argument))
                        input_dict[argument] = new Value(CNTKBackend.In(feed_dict[argument]));
                    else
                        throw new Exception($"CNTK backend: argument {argument.Name} is not found in inputs. Please double check the model and inputs in 'train_function'.");
                }

                var result = this.trainer.TrainMinibatch(input_dict, this.trainer_output);

                foreach (Variable o in this.trainer_output.Keys)
                    updated.Add(c.Out(this.trainer_output[o]));
            }

            if (this.metrics_func != null)
            {
                var input_dict = new Dictionary<Variable, Value>();
                foreach (Variable argument in this.metrics_func.Arguments)
                {
                    if (feed_dict.ContainsKey(argument))
                        input_dict[argument] = new Value(CNTKBackend.In(feed_dict[argument]));
                    else
                        throw new Exception($"CNTK backend: metrics argument {argument.Name} is not found in inputs. Please double check the model and inputs.");
                }

                var output_values = new Dictionary<Variable, Value>();
                foreach (Variable variable in this.metrics_outputs)
                    output_values[variable] = null;

                this.metrics_func.Evaluate(input_dict, output_values, DeviceDescriptor.CPUDevice);

                foreach (Variable o in this.metrics_outputs)
                {
                    Value value = output_values[o];
                    var v = c.Out(value);
                    updated.Add(v);
                }
            }

            if (this.unrelated_updates != null)
            {
                var input_dict = new Dictionary<Variable, Value>();
                foreach (Variable argument in this.unrelated_updates.Arguments)
                {
                    if (feed_dict.ContainsKey(argument))
                        input_dict[argument] = new Value(CNTKBackend.In(feed_dict[argument]));
                    else
                        throw new Exception($"CNTK backend: assign ops argument {argument.Name} is not found in inputs. Please double check the model and inputs.");
                }

                var output_values = new Dictionary<Variable, Value>();
                foreach (Variable variable in this.unrelated_updates.Inputs)
                    output_values[variable] = null;
                this.unrelated_updates.Evaluate(input_dict, output_values, DeviceDescriptor.CPUDevice);
            }

            return updated;
        }
    }
}