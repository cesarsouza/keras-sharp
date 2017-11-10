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

using Accord;
using Accord.Math;
using KerasSharp.Engine.Topology;
using System;
using System.Collections.Generic;

namespace KerasSharp.Models
{
    public class BaseLogger : Callback
    {
        public int seen;
        private Dictionary<string, double> totals;

        public BaseLogger()
        {
        }

        public override void on_batch_end(Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            int batch_size = (int)logs.get("size", 0);
            this.seen += batch_size;

            foreach (var item in logs)
            {
                var k = item.Key;
                double v = item.Value.To<double>();

                if (this.totals.ContainsKey(k))
                    this.totals[k] = totals[k] = v * batch_size;
                else
                    this.totals[k] = v * batch_size;
            }
        }

        public override void on_epoch_begin(int epoch, Dictionary<string, object> logs)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/callbacks.py#L207
            this.seen = 0;
            this.totals = new Dictionary<string, double>();
        }

        public override void on_epoch_end(int epoch, Dictionary<string, object> logs)
        {
            if (logs != null)
            {
                foreach (string k in (IEnumerable<string>)this.parameters["metrics"])
                {
                    if (this.totals.ContainsKey(k))
                    {
                        // Make value available to next callbacks.
                        logs[k] = this.totals[k] / (double)this.seen;
                    }
                }
            }
        }

    }
}