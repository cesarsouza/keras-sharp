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

namespace KerasSharp.Models
{
    public class ProgbarLogger : Callback
    {
        private int verbose;
        private object epochs;
        private bool use_steps;
        private int target;
        private Progbar progbar;
        private int seen;
        private List<(string, object)> log_values;

        public ProgbarLogger()
        {
        }

        public ProgbarLogger(string count_mode = "samples")
        {
            if (count_mode == "samples")
                this.use_steps = false;
            else if (count_mode == "steps")
                this.use_steps = true;
            else
                throw new ArgumentException("Unknown 'count_mode': " + count_mode);
        }

        public override void on_train_begin(Dictionary<string, object> logs = null)
        {
            this.verbose = (int)base.parameters["verbose"];
            this.epochs = base.parameters["epochs"];
        }

        public override void on_batch_begin(Dictionary<string, object> logs)
        {
            if (this.seen < this.target)
                this.log_values = new List<(string, object)>();
        }

        public override void on_batch_end(Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();
            int batch_size = (int)logs.get("size", 0);
            if (this.use_steps)
                this.seen += 1;
            else
                this.seen += batch_size;

            foreach (string k in (IEnumerable<string>)this.parameters["metrics"])
            {
                if (logs.ContainsKey(k))
                    this.log_values.Add((k, logs[k]));
            }

            // Skip progbar update for the last batch;
            // will be handled by on_epoch_end.
            if (this.verbose > 0 && this.seen < this.target)
                this.progbar.update(this.seen, this.log_values);
        }

        public override void on_epoch_begin(int epoch, Dictionary<string, object> logs)
        {
            if (this.verbose > 0)
            {
                Console.WriteLine($"Epoch {epoch + 1} / {this.epochs}");

                if (this.use_steps)
                    this.target = (int)this.parameters["steps"];
                else
                    this.target = (int)this.parameters["samples"];

                this.progbar = new Progbar(target: this.target, verbose: this.verbose);
            }

            this.seen = 0;
        }

        public override void on_epoch_end(int epoch, Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();
            foreach (string k in (IEnumerable<string>)this.parameters["metrics"])
            {
                if (logs.ContainsKey(k))
                    this.log_values.Add((k, logs[k]));
            }

            if (this.verbose > 0)
                this.progbar.update(seen, this.log_values, force: true);
        }

    }
}