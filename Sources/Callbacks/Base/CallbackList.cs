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
    public class CallbackList : List<Callback>
    {
        /// <summary>
        ///   Called at the beginning of training.
        /// </summary>
        /// 
        /// <param name="logs">Dictionary of logs.</param>
        /// 
        internal void on_train_begin(Dictionary<string, object> logs = null)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_train_begin(logs);
        }

        internal void on_epoch_begin(int epoch, Dictionary<string, object> logs = null)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_epoch_begin(epoch, logs);
        }

        internal void on_batch_begin(int batch_index, Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_batch_begin(logs);
        }

        internal void on_batch_end(int batch_index, Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_batch_end(logs);
        }

        internal void on_epoch_end(int epoch, Dictionary<string, object> logs)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_epoch_end(epoch, logs);
        }

        internal void on_train_end(Dictionary<string, object> logs = null)
        {
            if (logs == null)
                logs = new Dictionary<string, object>();

            foreach (Callback callback in this)
                callback.on_train_end(logs);
        }

        internal void set_model(object model)
        {
            foreach (Callback callback in this)
                callback.set_model(model);
        }

        internal void set_params(Dictionary<string, object> dictionary)
        {
            foreach (Callback callback in this)
                callback.set_params(dictionary);
        }
    }
}