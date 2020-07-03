"""Layer class. Contains the layer class and relevant methods.
Defines each layer of a neural network."""
import keras as K
import numpy.random as rand

import params

rand.seed(42)


def create_layer(layer_in, ltype):
    """Returns a new layer object of 'ltype' which has 'layer_in' as input."""
    if ltype == "conv":
        n_channels = layer_in.dims_out()[-1]
        return ConvLayer(layer_in, n_channels=n_channels)
    if ltype == "FC":
        n_channels = layer_in.dims_out()[-1]
        return DenseLayer(layer_in, n_channels=n_channels)
    if ltype == "2x2pool":
        return LocalPoolingLayer(layer_in)
    if ltype == "mbconv":
        n_channels = layer_in.dims_out()[-1]
        return MBConvLayer(layer_in, n_channels=n_channels)
    if ltype == "global_pool":
        return GlobalPoolingLayer(layer_in)
    raise Exception("Layertype " + str(ltype) + " should be one of param.LAYER_TYPE_OPTIONS")


def create_merge_layer(layer1_in, layer2_in):
    """Returns a new merge layer merging the outputs of 'layer1_in' and 'layer2_in'."""
    return MergeLayer(layer1_in, layer2_in)


def suitable_output_ltypes(ltype):
    """Returns suitable 'ltypes' for the output layer of a layer of 'ltype'."""
    if ltype == "conv":
        return ["mbconv", "2x2pool"]
    if ltype == "mbconv":
        return ["mbconv", "2x2pool", "merge", "global_pool"]
    if ltype == "2x2pool":
        return ["mbconv", "merge"]
    if ltype == "merge":
        return ["mbconv", "2x2pool", "merge"]
    if ltype == "global_pool":
        return ["FC"]
    if ltype == "FC":
        return []
    raise Exception("Layertype " + str(ltype) + " should be one of param.LAYER_TYPE_OPTIONS")


def suitable_input_ltypes(ltype):
    """Returns suitable 'ltypes' for the input layer of a layer of 'ltype'."""
    if ltype == "conv":
        return []
    if ltype == "mbconv":
        return ["mbconv", "conv", "2x2pool", "merge"]
    if ltype == "merge":
        return ["mbconv", "2x2pool", "merge"]
    if ltype == "2x2pool":
        return ["mbconv", "merge", "conv"]
    if ltype == "global_pool":
        return ["mbconv", "merge"]
    if ltype == "FC":
        return ["global_pool"]
    raise Exception("Layertype " + str(ltype) + " should be one of param.LAYER_TYPE_OPTIONS")


class Layer:
    """Defines a layer, the building block of a neural network."""

    def __init__(self, layer_in):
        self.ltype = "Undefined"
        if layer_in is None:  # The first layer has None for layer_in
            self.dims_in = [params.INPUT_IMG_SIZE, params.INPUT_IMG_SIZE, 3]
        else:
            self.dims_in = layer_in.dims_out()
        self.lsubtype = ""
        self.layer_in = layer_in
        self.is_mutable = False
        self.keras_tensor = "Not initialized"

    def print(self, consise=False):
        """Prints attributes of the layer."""
        if consise:
            print(', '.join("%0s: %8s" % item for item in
                            [self.ltype,
                             self.dims_in,
                             self.keras_tensor,
                             self.layer_in.keras_tensor]))
        else:
            print(', '.join("%0s: %8s" % item for item in vars(self).items()))

    def dims_out(self):
        """Returns output dimensions. Override in subclass."""

    def to_keras(self, in_tensor):
        """Returns a keras tensor. Override in subclass."""

    def deepcopy(self):
        """Returns a deepcopy of the instance."""
        lay_copy = self.__class__(layer_in=None)
        lay_copy.__dict__ = self.__dict__.copy()
        return lay_copy


class ConvLayer(Layer):
    """Convolution layer class. Can be either a regular convolution or a seperable convolution."""

    def __init__(self, layer_in, lsubtype="sepconv", n_channels=64,
                 kernel_size=3, stride=1, dilation_rate=1, repeats=1):
        super().__init__(layer_in)
        self.ltype = "conv"
        self.lsubtype = lsubtype
        self.is_mutable = True
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.repeats = repeats

    def mutate(self):
        """Mutates layer attributes."""
        options = ["sepconv", "conv"]
        self.lsubtype = rand.choice(options)

        if rand.rand() < 0.5:
            self.n_channels = min(512, self.n_channels * 2)
        else:
            self.n_channels = max(16, self.n_channels // 2)

        if rand.rand() < 0.5:
            self.kernel_size = min(7, self.kernel_size + 2)
        else:
            self.kernel_size = max(1, self.kernel_size - 2)

        if rand.rand() < 0.5 and self.dilation_rate == 1:
            self.stride = 2
        else:
            self.stride = 1

        if rand.rand() < 0.5 and self.stride == 1:
            self.dilation_rate = 2
        else:
            self.dilation_rate = 1

        if rand.rand() < 0.5:
            self.repeats += 1
        else:
            self.repeats = max(1, self.repeats - 1)

    def dims_out(self):
        resolution_out = self.dims_in[0] // (self.stride ** self.repeats)
        return [resolution_out, resolution_out, self.n_channels]

    def to_keras(self, in_tensor):
        tensor = in_tensor
        for _ in range(self.repeats):
            if self.lsubtype == "conv":
                tensor = K.layers.Convolution2D(filters=self.n_channels,
                                                kernel_size=self.kernel_size,
                                                strides=self.stride,
                                                padding='same',
                                                kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                                dilation_rate=self.dilation_rate)(tensor)
            else:
                tensor = K.layers.SeparableConv2D(filters=self.n_channels,
                                                  kernel_size=self.kernel_size,
                                                  strides=self.stride,
                                                  padding='same',
                                                  kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                                  dilation_rate=self.dilation_rate)(tensor)
            tensor = K.layers.ReLU()(tensor)
            tensor = K.layers.BatchNormalization()(tensor)
        return tensor


class DenseLayer(Layer):
    """Dense layer class. Also known as a fully connected layer."""

    def __init__(self, layer_in, n_channels=64, repeats=1, final_layer=False):
        super().__init__(layer_in)
        self.ltype = "FC"
        self.is_mutable = True
        self.n_channels = n_channels
        self.repeats = repeats
        self.final_layer = final_layer

    def mutate(self):
        """Mutates layer attributes."""
        if rand.rand() < 0.5:
            self.n_channels = min(512, self.n_channels * 2)
        else:
            self.n_channels = max(16, self.n_channels // 2)

        if rand.rand() < 0.5:
            self.repeats += 1
        else:
            self.repeats = max(1, self.repeats - 1)

    def dims_out(self):
        if len(self.dims_in) > 1:
            return [self.dims_in[0], self.dims_in[1], self.n_channels]
        return [self.n_channels]

    def to_keras(self, in_tensor):
        tensor = in_tensor
        for _ in range(self.repeats):
            tensor = K.layers.Dense(kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                    units=self.n_channels)(tensor)
            if self.final_layer:
                tensor = K.layers.Softmax()(tensor)
            else:
                tensor = K.layers.ReLU()(tensor)
                tensor = K.layers.BatchNormalization()(tensor)
        return tensor


class LocalPoolingLayer(Layer):
    """Local pooling layer class. Performs 2x2 pooling."""

    def __init__(self, layer_in, lsubtype="maxpool"):
        super().__init__(layer_in)
        self.ltype = "2x2pool"
        self.lsubtype = lsubtype

    def mutate(self):
        """Mutates layer attributes."""
        options = ["maxpool", "avgpool"]
        options.remove(self.lsubtype)
        self.lsubtype = rand.choice(options)

    def dims_out(self):
        resolution_out = self.dims_in[0] // 2
        return [resolution_out, resolution_out, self.dims_in[2]]

    def to_keras(self, in_tensor):
        if self.lsubtype == "maxpool":
            return K.layers.MaxPooling2D(pool_size=2)(in_tensor)
        return K.layers.AveragePooling2D(pool_size=2)(in_tensor)


class GlobalPoolingLayer(Layer):
    """Global pooling layer class. Reduces width x height dimensions to 1x1 by averaging."""

    def __init__(self, layer_in, lsubtype="avgpool"):
        super().__init__(layer_in)
        self.ltype = "global_pool"
        self.lsubtype = lsubtype

    def mutate(self):
        """Mutates layer attributes."""
        options = ["maxpool", "avgpool"]
        options.remove(self.lsubtype)
        self.lsubtype = rand.choice(options)

    def dims_out(self):
        return [self.dims_in[2]]

    def to_keras(self, in_tensor):
        if self.lsubtype == "maxpool":
            return K.layers.GlobalMaxPooling2D()(in_tensor)
        return K.layers.GlobalAveragePooling2D()(in_tensor)


class MergeLayer(Layer):
    """Merge layer class. Merges output of two input layers by elementwise addition
    in case of equal input dimensions or concatanation otherwise."""

    def __init__(self, layer1_in, layer2_in):
        super().__init__(layer1_in)
        self.ltype = "merge"
        self.lsubtype = "add"
        self.layer2_in = layer2_in

    def update_mergetype(self):
        """Updates whether the merge is performed by elementwise addition or concatanation."""
        if self.layer_in.dims_out()[-1] == self.layer2_in.dims_out()[-1]:
            self.lsubtype = "add"
        else:
            self.lsubtype = "concat"

    def dims_out(self):
        self.update_mergetype()
        if self.lsubtype == "add":
            return self.dims_in
        if len(self.dims_in) == 3:
            return [self.dims_in[0],
                    self.dims_in[1],
                    self.dims_in[2] + self.layer2_in.dims_out()[2]]
        return [self.dims_in[0] + self.layer2_in.dims_out()[0]]

    def to_keras(self, in_tensor):
        self.update_mergetype()
        if self.lsubtype == "add":
            return K.layers.Add()(in_tensor)
        return K.layers.Concatenate()(in_tensor)

    def deepcopy(self):
        """Returns a deepcopy of the instance."""
        lay_copy = self.__class__(layer1_in=None, layer2_in=None)
        lay_copy.__dict__ = self.__dict__.copy()
        return lay_copy


class MBConvLayer(Layer):
    """Mobile Inverted Bottleneck Block layer class. See https://arxiv.org/abs/1801.04381."""

    def __init__(self, layer_in, n_channels=64, depth_multiplier=3,
                 kernel_size=3, dilation_rate=1, has_se=True, repeats=1):
        super().__init__(layer_in)
        self.ltype = "mbconv"
        self.is_mutable = True
        self.n_channels = n_channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.has_se = has_se
        self.repeats = repeats

    def mutate(self):
        """Mutates layer attributes."""
        if rand.rand() < 0.5:
            self.n_channels = min(512, self.n_channels * 2)
        else:
            self.n_channels = max(16, self.n_channels // 2)

        if rand.rand() < 0.5:
            self.kernel_size = min(7, self.kernel_size + 2)
        else:
            self.kernel_size = max(1, self.kernel_size - 2)

        if rand.rand() < 0.5:
            self.dilation_rate = 2
        else:
            self.dilation_rate = 1

        if rand.rand() < 0.5:
            self.has_se = True
        else:
            self.has_se = False

        if rand.rand() < 0.5:
            self.repeats += 1
        else:
            self.repeats = max(1, self.repeats - 1)

        if rand.rand() < 0.5:
            self.depth_multiplier += 1
        else:
            self.depth_multiplier = max(1, self.depth_multiplier - 1)

    def dims_out(self):
        resolution_out = self.dims_in[0]
        return [resolution_out, resolution_out, self.n_channels]

    def max_params(self):
        """Returns the total number of parameters in the expanded depthwise convolution."""
        return self.dims_in[0] * self.dims_in[1] * self.n_channels * self.depth_multiplier

    def to_keras(self, in_tensor):
        tensor = in_tensor
        for rep in range(self.repeats):
            residual_tensor = tensor

            # Expansion layer (increase channels)
            tensor = K.layers.Convolution2D(filters=self.n_channels * self.depth_multiplier,
                                            kernel_size=1,
                                            kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                            padding='same')(residual_tensor)
            tensor = K.layers.BatchNormalization()(tensor)
            tensor = K.layers.ReLU()(tensor)

            # Depthwise convolution layer
            tensor = K.layers.DepthwiseConv2D(kernel_size=self.kernel_size,
                                              padding='same',
                                              kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                              dilation_rate=self.dilation_rate)(tensor)
            tensor = K.layers.BatchNormalization()(tensor)
            expanded_tensor = K.layers.ReLU()(tensor)

            # Add Squeeze-and-Excitation module
            if self.has_se:
                se_tensor = K.layers.GlobalAveragePooling2D()(expanded_tensor)
                se_tensor = K.layers.Reshape((1, 1, self.n_channels * self.depth_multiplier))(se_tensor)
                reduced_n_channels = max(1,
                                         self.n_channels * self.depth_multiplier // 4)  # Fixed reduction ratio of 0.25
                se_tensor = K.layers.Dense(kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                           units=reduced_n_channels)(se_tensor)
                se_tensor = K.layers.ReLU()(se_tensor)
                se_tensor = K.layers.Dense(kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                           units=self.n_channels * self.depth_multiplier)(se_tensor)
                se_tensor = K.layers.Activation('sigmoid')(se_tensor)
                expanded_tensor = K.layers.Multiply()([tensor, se_tensor])

            # Projection layer (decrease channels)
            tensor = K.layers.Convolution2D(filters=self.n_channels,
                                            kernel_size=1,
                                            kernel_regularizer=K.regularizers.l2(params.WEIGHT_DECAY),
                                            padding='same')(expanded_tensor)
            tensor = K.layers.BatchNormalization()(tensor)

            # Add skip connection
            if (rep == 0 and self.dims_in[-1] == self.n_channels) or rep > 0:
                tensor = K.layers.Add()([residual_tensor, tensor])
        return tensor

#
# class ActivationLayer(Layer):
#     def __init__(self, layer_in, lsubtype="ReLU"):
#         super().__init__(layer_in)
#         self.ltype = "act"
#         self.lsubtype = lsubtype
#
#     def mutate(self):
#         options = ["ReLU", "elu", "leakyReLU", "selu"]
#         if self.lsubtype in options:
#             options.remove(self.lsubtype)
#         self.lsubtype = rand.choice(options)
#
#     def dims_out(self):
#         return self.dims_in

# def to_keras(self):
#     if self.lsubtype == "ReLU":
#         return K.layers.ReLU()
#     if self.lsubtype == "sigmoid":
#         return K.layers.Activation('sigmoid')
#     if self.lsubtype == "softmax":
#         return K.layers.Softmax()
#     if self.lsubtype == "elu":
#         return K.layers.ELU()
#     if self.lsubtype == "leakyReLU":
#         return K.layers.LeakyReLU()
#     if self.lsubtype == "selu":
#         return K.layers.Activation("selu")
#     raise Exception("Layersubtype should be one of param.ACTIVATION_LAYER_OPTIONS")
#
# def to_keras(self, in_tensor):
#     if self.lsubtype == "ReLU":
#         return K.layers.ReLU()(in_tensor)
#     if self.lsubtype == "sigmoid":
#         return K.layers.Activation('sigmoid')(in_tensor)
#     if self.lsubtype == "softmax":
#         return K.layers.Softmax()(in_tensor)
#     raise Exception("Layersubtype should be one of param.ACTIVATION_LAYER_OPTIONS")

# class BatchNormLayer(Layer):
#     def __init__(self, layer_in):
#         super().__init__(layer_in)
#         self.ltype = "BN"
#
#     def dims_out(self):
#         return self.dims_in
#
#     # def to_keras(self):
#     #     return K.layers.BatchNormalization()
#
#     def to_keras(self, in_tensor):
#         return K.layers.BatchNormalization()(in_tensor)

# class FlattenLayer(Layer):
#     def __init__(self, layer_in):
#         super().__init__(layer_in)
#         self.ltype = "flatten"
#
#     def dims_out(self):
#         if len(self.dims_in) > 1:
#             return [self.dims_in[0] * self.dims_in[1] * self.dims_in[2]]
#         return self.dims_in
#
#     # def to_keras(self):
#     #     if len(self.dims_in) > 1:
#     #         return K.layers.Flatten()
#     #     return K.layers.Lambda(lambda x: x)
#
#     def to_keras(self, in_tensor):
#         if len(self.dims_in) > 1:
#             return K.layers.Flatten()(in_tensor)
#         return K.layers.Lambda(lambda x: x)(in_tensor)

# class DropoutLayer(Layer):
#     def __init__(self, layer_in, rate=0.2):
#         super().__init__(layer_in)
#         self.ltype = "dropout"
#         self.rate = rate
#
#     def mutate(self):
#         if rand.rand() < 0.5:
#             self.rate = min(0.5, self.rate * 2.)
#         else:
#             self.rate /= 2
#
#     def dims_out(self):
#         return self.dims_in
#
#     # def to_keras(self):
#     #     return K.layers.Dropout(self.rate)
#
#     def to_keras(self, in_tensor):
#         return K.layers.Dropout(self.rate)(in_tensor)
