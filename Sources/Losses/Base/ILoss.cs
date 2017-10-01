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

namespace KerasSharp.Losses
{
    using KerasSharp.Engine.Topology;
    

    /// <summary>
    ///   Common interface for loss functions.
    /// </summary>
    /// 
    public interface ILoss
    {
        /// <summary>
        ///   Wires the given ground-truth and predictions through the desired loss.
        /// </summary>
        /// 
        /// <param name="expected">The ground-truth data that the model was supposed to approximate.</param>
        /// <param name="actual">The actual data predicted by the model.</param>
        /// 
        /// <returns>A scalar value representing how far the model's predictions were from the ground-truth.</returns>
        /// 
        Tensor Call(Tensor expected, Tensor actual, Tensor sample_weight = null, Tensor mask = null);
    }

    // Aliases
    public sealed class MSE : MeanSquareError { }
    public sealed class MAE : MeanAbsoluteError { }
    public sealed class MAPE : MeanAbsolutePercentageError { }
    public sealed class MSLE : MeanSquareLogarithmicError { }
    public sealed class Cosine : CosineProximity { }
}