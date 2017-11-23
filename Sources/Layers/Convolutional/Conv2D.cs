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

namespace KerasSharp
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    using System.Runtime.Serialization;
    using KerasSharp.Constraints;
    using KerasSharp.Regularizers;
    using KerasSharp.Initializers;
    using Accord.Math;
    using KerasSharp.Engine.Topology;

    using static KerasSharp.Backends.Current;

    /// <summary>
    ///   Abstract nD convolution layer (private, used as implementation base).
    ///   This layer creates a convolution kernel that is convolved
    ///   with the layer input to produce a tensor of outputs.
    ///   If `use_bias` is True, a bias vector is created and added to the outputs.
    ///   Finally, if `activation` is not `None`,
    ///   it is applied to the outputs as well.
    /// </summary>
    /// 
    public class _Conv : Layer
    {
        private int rank;
        private int filters;
        private int[] kernel_size;
        private int[] strides;
        private PaddingType padding;
        private DataFormatType? data_format;
        private int[] dilation_rate;
        private IActivationFunction activation;
        private bool use_bias;
        private IWeightInitializer kernel_initializer;
        private IWeightInitializer bias_initializer;
        private IWeightRegularizer kernel_regularizer;
        private IWeightRegularizer bias_regularizer;
        private IWeightConstraint kernel_constraint;
        private IWeightConstraint bias_constraint;
        private Tensor kernel;
        private Tensor bias;

        /// <summary>
        /// Initializes a new instance of the <see cref="_Conv" /> class.
        /// </summary>
        /// <param name="rank">rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.</param>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e.the number output of filters in the convolution).</param>
        /// <param name="kernel_size">An integer or tuple/list of n integers, specifying the dimensions of the convolution window.</param>
        /// <param name="strides">An integer or tuple/list of n integers, specifying the strides of the convolution. Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.</param>
        /// <param name="padding">One of `"valid"` or `"same"` (case-insensitive).</param>
        /// <param name="data_format">A string, one of `channels_last` (default) or `channels_first`. The ordering of the dimensions in the inputs. 
        ///   `channels_last` corresponds to inputs with shape `(batch, ..., channels)` while `channels_first` corresponds to inputs with shape 
        ///   `(batch, channels, ...)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. 
        ///   If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate">An integer or tuple/list of n integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any `dilation_rate` value != 1 is incompatible with specifying any `strides` value != 1.</param>
        /// <param name="activation">Activation function to use (see[activations](../activations.md)). If you don't specify anything, no activation is applied (ie. "linear" activation: `a(x) = x`).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the `kernel` weights matrix (see[initializers](../initializers.md)).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see[initializers](../initializers.md)).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the `kernel` weights matrix (see[regularizer](../regularizers.md)).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see[regularizer](../regularizers.md)).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer(its "activation"). (see[regularizer](../regularizers.md)).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel matrix (see[constraints](../constraints.md)).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see[constraints](../constraints.md)).</param>
        public _Conv(int rank,
                         int filters,
                         int[] kernel_size,
                         int[] strides = null,
                         PaddingType padding = PaddingType.Valid,
                         DataFormatType? data_format = null,
                         int[] dilation_rate = null,
                         IActivationFunction activation = null,
                         bool use_bias = true,
                         IWeightInitializer kernel_initializer = null,
                         IWeightInitializer bias_initializer = null,
                         IWeightRegularizer kernel_regularizer = null,
                         IWeightRegularizer bias_regularizer = null,
                         IWeightRegularizer activity_regularizer = null,
                         IWeightConstraint kernel_constraint = null,
                         IWeightConstraint bias_constraint = null,
                         int?[] input_shape = null)
            : base(input_shape: input_shape)
        {
            if (kernel_initializer == null)
                kernel_initializer = new GlorotUniform();

            if (bias_initializer == null)
                bias_initializer = new Zeros();

            if (strides == null)
                strides = Vector.Create<int>(size: rank, value: 1);

            if (dilation_rate == null)
                dilation_rate = Vector.Create<int>(size: rank, value: 1);

            if (data_format == null)
                data_format = K.image_data_format();

            if (kernel_size.Length != rank)
                throw new ArgumentException("kernel_size");

            if (strides.Length != rank)
                throw new ArgumentException("strides");

            if (dilation_rate.Length != rank)
                throw new ArgumentException("dilation_rate");

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/convolutional.py#L101

            this.rank = rank;
            this.filters = filters;
            this.kernel_size = kernel_size;
            this.strides = strides;
            this.padding = padding;
            this.data_format = data_format;
            this.dilation_rate = dilation_rate;
            this.activation = activation;
            this.use_bias = use_bias;
            this.kernel_initializer = kernel_initializer;
            this.bias_initializer = bias_initializer;
            this.kernel_regularizer = kernel_regularizer;
            this.bias_regularizer = bias_regularizer;
            this.activity_regularizer = activity_regularizer;
            this.kernel_constraint = kernel_constraint;
            this.bias_constraint = bias_constraint;
            this.input_spec = new List<InputSpec> { new InputSpec(ndim: this.rank + 2) };
        }

        protected override void build(List<int?[]> input_shapes)
        {
            if (input_shapes.Count > 1)
                throw new Exception();

            var input_shape = input_shapes[0];

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/convolutional.py#L119

            int channel_axis;
            if (this.data_format == DataFormatType.ChannelsFirst)
                channel_axis = 1;
            else
                channel_axis = -1;

            if (input_shape.Get(channel_axis) == null)
                throw new Exception("The channel dimension of the inputs should be defined. Found `None`.");

            int input_dim = input_shape.Get(channel_axis).Value;
            int[] kernel_shape = this.kernel_size.Concat(new[] { input_dim, this.filters }).ToArray();

            this.kernel = this.add_weight(shape: kernel_shape,
                                      initializer: this.kernel_initializer,
                                      name: "kernel",
                                      regularizer: this.kernel_regularizer,
                                      constraint: this.kernel_constraint);
            if (this.use_bias)
            {
                this.bias = this.add_weight(shape: new int[] { this.filters },
                                            initializer: this.bias_initializer,
                                            name: "bias",
                                            regularizer: this.bias_regularizer,
                                            constraint: this.bias_constraint);
            }
            else
            {
                this.bias = null;
            }

            // Set input spec.
            this.input_spec = new List<InputSpec> { new InputSpec(ndim: this.rank + 2, axes: new Dictionary<int, int> { { channel_axis, input_dim } }) };
            this.built = true;
        }

        protected override Tensor InnerCall(Tensor inputs, Tensor mask = null, bool? training = null)
        {
            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/convolutional.py#L149

            if (mask != null)
                throw new Exception();

            if (training != null)
                throw new Exception();

            Tensor outputs = null;

            if (this.rank == 1)
            {
                outputs = K.conv1d(
                    inputs,
                    this.kernel,
                    strides: this.strides[0],
                    padding: this.padding,
                    data_format: this.data_format,
                    dilation_rate: this.dilation_rate[0]);
            }
            if (this.rank == 2)
            {
                outputs = K.conv2d(
                    inputs,
                    this.kernel,
                    strides: this.strides,
                    padding: this.padding,
                    data_format: this.data_format,
                    dilation_rate: this.dilation_rate);
            }
            if (this.rank == 3)
            {
                outputs = K.conv3d(
                    inputs,
                    this.kernel,
                    strides: this.strides,
                    padding: this.padding,
                    data_format: this.data_format,
                    dilation_rate: this.dilation_rate);
            }
            if (this.use_bias)
            {
                outputs = K.bias_add(
                    outputs,
                    this.bias,
                    data_format: this.data_format);
            }

            if (this.activation != null)
                return this.activation.Call(outputs, null);
            return outputs;
        }

        public override List<int?[]> compute_output_shape(List<int?[]> input_shapes)
        {
            if (input_shapes.Count != 1)
                throw new Exception("Expected a single input.");
            int?[] input_shape = input_shapes[0];

            // https://github.com/fchollet/keras/blob/f65a56fb65062c8d14d215c9f4b1015b97cc5bf3/keras/layers/convolutional.py#L185

            if (this.data_format == DataFormatType.ChannelsLast)
            {
                var space = input_shape.Get(1, -1);
                var new_space = new List<int?>();
                for (int i = 0; i < space.Length; i++)
                {
                    int? new_dim = conv_utils.conv_output_length(
                        space[i],
                        this.kernel_size[i],
                        padding: this.padding,
                        stride: this.strides[i],
                        dilation: this.dilation_rate[i]);
                    new_space.Add(new_dim);
                }

                return new[] { new[] { input_shape[0] }.Concat(new_space).Concat(new int?[] { this.filters }).ToArray() }.ToList();
            }
            else if (this.data_format == DataFormatType.ChannelsFirst)
            {
                var space = input_shape.Get(2, 0);
                var new_space = new List<int?>();
                for (int i = 0; i < space.Length; i++)
                {
                    int? new_dim = conv_utils.conv_output_length(
                        space[i],
                        this.kernel_size[i],
                        padding: this.padding,
                        stride: this.strides[i],
                        dilation: this.dilation_rate[i]);
                    new_space.Add(new_dim);
                }

                return new[] { new[] { input_shape[0] }.Concat(new int?[] { this.filters }).Concat(new_space).ToArray() }.ToList();
            }
            else
            {
                throw new Exception();
            }
        }

        //override Dictionary<string, object> get_config()
        //{
        //    return new Dictionary<string, object>
        //    {
        //        { "rank",  this.rank },
        //        { "filters",  this.filters },
        //        { "kernel_size",  this.kernel_size },
        //        { "strides",  this.strides },
        //        { "padding",  this.padding },
        //        { "data_format",  this.data_format },
        //        { "dilation_rate",  this.dilation_rate },
        //        { "activation",  activations.serialize(this.activation) },
        //        { "use_bias",  this.use_bias },
        //        { "kernel_initializer",  initializers.serialize(this.kernel_initializer) },
        //        { "bias_initializer",  initializers.serialize(this.bias_initializer) },
        //        { "kernel_regularizer",  regularizers.serialize(this.kernel_regularizer) },
        //        { "bias_regularizer",  regularizers.serialize(this.bias_regularizer) },
        //        { "activity_regularizer",  regularizers.serialize(this.activity_regularizer) },
        //        { "kernel_constraint",  constraints.serialize(this.kernel_constraint) },
        //        { "bias_constraint",  constraints.serialize(this.bias_constraint) },
        //    };

        //base_config = super(_Conv, self).get_config()
        //    return dict(list(base_config.items()) + list(config.items()))
    }


    /// <summary>
    ///   2D convolution layer (e.g. spatial convolution over images).
    /// </summary>
    /// 
    /// <remarks>
    ///   This layer creates a convolution kernel that is convolved
    ///   with the layer input to produce a tensor of
    ///   outputs.If `use_bias` is True,
    ///   a bias vector is created and added to the outputs.Finally, if
    ///   `activation` is not `None`, it is applied to the outputs as well.
    ///   When using this layer as the first layer in a model,
    ///   provide the keyword argument `input_shape`
    ///   (tuple of integers, does not include the sample axis),
    ///   e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    ///   in `data_format="channels_last"`.
    /// </remarks>
    /// 
    /// <seealso cref="KerasSharp.Engine.Topology.Layer" />
    /// 
    [DataContract]
    public class Conv2D : _Conv
    {

        public Conv2D(int filters,
                 int[] kernel_size = null,
                 int[] strides = null,
                 PaddingType padding = PaddingType.Valid,
                 DataFormatType? data_format = null,
                 int[] dilation_rate = null,
                 IActivationFunction activation = null,
                 bool use_bias = true,
                 IWeightInitializer kernel_initializer = null,
                 IWeightInitializer bias_initializer = null,
                 IWeightRegularizer kernel_regularizer = null,
                 IWeightRegularizer bias_regularizer = null,
                 IWeightRegularizer activity_regularizer = null,
                 IWeightConstraint kernel_constraint = null,
                 IWeightConstraint bias_constraint = null,
                 int?[] input_shape = null)
         : base(rank: 2,
            filters: filters,
            kernel_size: kernel_size,
            strides: strides,
            padding: padding,
            data_format: data_format,
            dilation_rate: dilation_rate,
            activation: activation,
            use_bias: use_bias,
            kernel_initializer: kernel_initializer,
            bias_initializer: bias_initializer,
            kernel_regularizer: kernel_regularizer,
            bias_regularizer: bias_regularizer,
            activity_regularizer: activity_regularizer,
            kernel_constraint: kernel_constraint,
            bias_constraint: bias_constraint,
            input_shape: input_shape)
        {
            this.input_spec = new List<InputSpec> { new InputSpec(ndim: 4) };
        }

        public Conv2D(int filters,
         int[] kernel_size = null,
         int[] strides = null,
         PaddingType padding = PaddingType.Valid,
         DataFormatType? data_format = null,
         int[] dilation_rate = null,
         string activation = null,
         bool use_bias = true,
         IWeightInitializer kernel_initializer = null,
         IWeightInitializer bias_initializer = null,
         IWeightRegularizer kernel_regularizer = null,
         IWeightRegularizer bias_regularizer = null,
         IWeightRegularizer activity_regularizer = null,
         IWeightConstraint kernel_constraint = null,
         IWeightConstraint bias_constraint = null,
         int?[] input_shape = null)
     : this(filters: filters,
        kernel_size: kernel_size,
        strides: strides,
        padding: padding,
        data_format: data_format,
        dilation_rate: dilation_rate,
        activation: Activation.Create(activation),
        use_bias: use_bias,
        kernel_initializer: kernel_initializer,
        bias_initializer: bias_initializer,
        kernel_regularizer: kernel_regularizer,
        bias_regularizer: bias_regularizer,
        activity_regularizer: activity_regularizer,
        kernel_constraint: kernel_constraint,
        bias_constraint: bias_constraint,
        input_shape: input_shape)
        {
        }

    }
}
