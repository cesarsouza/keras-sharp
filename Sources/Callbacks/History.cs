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

using System.Collections.Generic;

namespace KerasSharp.Models
{
    /// <summary>
    ///   Callback that records events into a `History` object. This callback 
    ///   is automatically applied to every Keras model.The `History` object 
    ///   gets returned by the `fit` method of models.
    /// </summary>
    /// 
    /// <seealso cref="KerasSharp.Models.Callback" />
    /// 
    public class History : Callback
    {
        private List<int> epoch;
        private Dictionary<string, List<object>> history;

        public override void on_batch_begin(Dictionary<string, object> logs)
        {
            this.epoch = new List<int>();
            this.history = new Dictionary<string, List<object>>();
        }

        public override void on_epoch_end(int epoch, Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            this.epoch.Add(epoch);

            foreach (var item in logs)
            {
                if (!this.history.ContainsKey(item.Key))
                    this.history[item.Key] = new List<object>();
                this.history[item.Key].Add(item.Value);
            }
        }

    }
}