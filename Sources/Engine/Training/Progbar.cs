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
using System;
using System.Collections.Generic;

namespace KerasSharp.Models
{
    internal class Progbar
    {
        private object p;
        private int verbose;
        private int target;
        private Dictionary<string, List<double>> sum_values;
        private List<string> unique_values;
        private DateTime start;
        private DateTime last_update;
        private double interval;
        private int total_width;
        private int seen_so_far;
        private int width;

        public Progbar(int? target, int width = 30, int verbose = 1, double interval = 0.05)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/utils/generic_utils.py#L219
            this.width = width;
            if (target == null)
                target = -1;
            this.target = target.Value;
            this.sum_values = new Dictionary<string, List<double>>();
            this.unique_values = new List<string>();
            this.start = DateTime.Now;
            this.last_update = DateTime.MinValue;
            this.interval = interval;
            this.total_width = 0;
            this.seen_so_far = 0;
            this.verbose = verbose;
        }

        public Progbar(int target)
        {
            this.target = target;
        }

        public void update(int current, List<(string, object)> values = null, bool force = false)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/utils/generic_utils.py#L241

            if (values == null)
                values = new List<(string, object)>();

            foreach (var (k, v) in values)
            {
                if (!this.sum_values.ContainsKey(k))
                {
                    this.sum_values[k] = new List<double>() { v.To<double>() * (current - this.seen_so_far), current - this.seen_so_far };
                    this.unique_values.Add(k);
                }
                else
                {
                    this.sum_values[k][0] += v.To<double>() * (current - this.seen_so_far);
                    this.sum_values[k][1] += (current - this.seen_so_far);
                }
            }
            this.seen_so_far = current;


            var now = DateTime.Now;
            if (this.verbose > 0)
            {
                if (!force && (now - this.last_update).TotalSeconds < this.interval)
                    return;
            }

            int prev_total_width = this.total_width;
            Console.Write(new String('\b', prev_total_width));
            Console.Write('\r');

            if (this.target != -1)
            {
                int numdigits = (int)(Math.Floor(Math.Log10(this.target))) + 1;
                string bar = $"{current}/{this.target} [";
                double prog = current / this.target;
                int prog_width = (int)(this.width * prog);
                if (prog_width > 0)
                {
                    bar += (new String('=', prog_width - 1));
                    if (current < this.target)
                        bar += '>';
                    else
                        bar += '=';
                }
                bar += (new String('.', this.width - prog_width));
                bar += ']';
                Console.Write(bar);
                this.total_width = bar.Length;
            }

            double time_per_unit;
            if (current != 0)
                time_per_unit = (now - this.start).TotalSeconds / (double)current;
            else
                time_per_unit = 0;
            double eta = time_per_unit * (this.target - current);
            string info = "";
            if (current < this.target && this.target != -1)
                info += $" - ETA: {eta}s";
            else
                info += $" - {now - this.start}s";
            foreach (string k in this.unique_values)
            {
                info += $" - {k}:";

                double avg = this.sum_values[k][0] / (double)Math.Max(1, this.sum_values[k][1]);
                info += $" {avg}";
            }

            this.total_width += info.Length;
            if (prev_total_width > this.total_width)
                info += new String(' ', prev_total_width - this.total_width);

            Console.Write(info);
            Console.Out.Flush();

            if (current >= this.target)
                Console.Write("\n");

            if (this.verbose == 2)
            {
                if (current >= this.target)
                {
                    info = $"{(now - this.start)}s";
                    foreach (string k in this.unique_values)
                    {
                        info += $" - {k}s:";
                        double avg = this.sum_values[k][0] / (double)Math.Max(1, this.sum_values[k][1]);
                        Console.Write(info + "\n");
                    }
                }
            }

            this.last_update = now;
        }
    }
}