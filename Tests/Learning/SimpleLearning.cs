using Accord.Math;
using KerasSharp;
using KerasSharp.Activations;
using KerasSharp.Initializers;
using KerasSharp.Losses;
using KerasSharp.Metrics;
using KerasSharp.Models;
using KerasSharp.Optimizers;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace Tests
{

    [TestFixture]
    public class SimpleLearningTest
    {
        static float[,] x =
        {
            { 0, 0 },
            { 0, 1 },
            { 1, 0 },
            { 1, 1 },
        };

        static float[] and = { 0, 0, 0, 1 };
        static float[] or  = { 0, 1, 1, 1 };
        static float[] nor = { 1, 0, 0, 0 };
        static float[] xor = { 0, 1, 1, 0 };

        private static IEnumerable<string> Backends()
        {
            yield return Setup.TENSORFLOW;
            //yield return new TestCaseData(new[] { Setup.CNTK });
        }

        private static IEnumerable<float[]> Targets()
        {
            yield return and;
            yield return or;
            yield return nor;
            yield return xor;
        }

        [Test, Combinatorial]
        public void mlp_should_learn_all(
            [ValueSource("Backends")] string backend, 
            [ValueSource("Targets")] float[] y,
            [Values(false, true)] bool useBias)
        {
            KerasSharp.Backends.Current.Switch(backend);

            var model = new Sequential();
            model.Add(new Dense(5, input_dim: 2,
                kernel_initializer: new GlorotUniform(),
                bias_initializer: new GlorotUniform(),
                use_bias: useBias,
                activation: new Sigmoid()));
            model.Add(new Dense(1,
                kernel_initializer: new GlorotUniform(),
                bias_initializer: new GlorotUniform(),
                use_bias: useBias,
                activation: new Sigmoid()));

            model.Compile(loss: new MeanSquareError(), optimizer: new SGD(lr: 1), metrics: new[] { new Accuracy() });

            model.fit(x, y, epochs: 1000, batch_size: y.Length);

            double[] pred = Matrix.Round(model.predict(x, batch_size: y.Length)[0].To<double[,]>()).GetColumn(0);

            Assert.AreEqual(y, pred);
        }

        [Test, Combinatorial]
        public void perceptron_should_learn_all_except_xor_nor(
            [ValueSource("Backends")] string backend,
            [ValueSource("Targets")] float[] y,
            [Values(false, true)] bool useBias)
        {
            KerasSharp.Backends.Current.Switch(backend);

            var model = new Sequential();
            model.Add(new Dense(1, input_dim: 2,
                kernel_initializer: new GlorotUniform(),
                bias_initializer: new GlorotUniform(),
                use_bias: useBias,
                activation: new Sigmoid()));

            model.Compile(loss: new MeanSquareError(), optimizer: new SGD(lr: 1), metrics: new[] { new Accuracy() });

            model.fit(x, y, epochs: 1000, batch_size: y.Length);

            Array yy = model.predict(x, batch_size: y.Length)[0];
            float[] pred = MatrixEx.Round(yy.To<float[,]>()).GetColumn(0);

            if ( (useBias && (y == xor)) || 
                (!useBias && (y == xor || y == nor || y == and)))
            {
                Assert.AreNotEqual(y, pred);
            }
            else
            {
                Assert.AreEqual(y, pred);
            }
        }


        [TearDown]
        public void TearDown()
        {
            KerasSharp.Backends.Current.K.clear_session();
        }
    }
}
