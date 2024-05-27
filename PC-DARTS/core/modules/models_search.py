import tensorflow as tf
from absl import logging
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Flatten, Conv2D, MaxPool2D,
                                     GlobalAveragePooling2D, Softmax)
from modules.operations import (OPS, FactorizedReduce, ReLUConvBN,
                                BatchNormalization, kernel_init, regularizer)
from modules.genotypes import PRIMITIVES, Genotype


def channel_shuffle(x, groups):
    _, h, w, num_channels = x.shape # batch_size, height, width, num_channels

    assert num_channels % groups == 0 # groups : K

    channels_per_group = num_channels // groups

    x = tf.reshape(x, [-1, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, h, w, num_channels])

    return x

class MixedOP(tf.keras.layers.Layer):
    """Mixed OP"""
    def __init__(self, ch, strides, wd, name='MixedOP', **kwargs):
        super(MixedOP, self).__init__(name=name, **kwargs)

        self._ops = []
        self.mp = MaxPool2D(2, strides=2, padding='valid')
        self.K = 4

        for primitive in PRIMITIVES:
            op = OPS[primitive](ch // self.K, strides, wd, False)

            # append batchnorm after pool layer
            if 'pool' in primitive:
                op = Sequential([op, BatchNormalization(affine=False)])

            self._ops.append(op)

    def call(self, x, weights):
        # channel proportion k
        x_1 = x[:, :, :, :x.shape[3] // self.K] # selected channels
        x_2 = x[:, :, :, x.shape[3] // self.K:] # masked channels

        # selected channels are sent into mixed computation of operations
        x_1 = tf.add_n([w * op(x_1) for w, op in
                        zip(tf.split(weights, len(PRIMITIVES)), self._ops)])

        # reduction cell needs pooling before concat
        if x_1.shape[2] == x.shape[2]:
            ans = tf.concat([x_1, x_2], axis=3)
        else:
            ans = tf.concat([x_1, self.mp(x_2)], axis=3)

        return channel_shuffle(ans, self.K)


class Cell(tf.keras.layers.Layer):
    """Cell Layer"""
    def __init__(self, steps, multiplier, ch, reduction, reduction_prev, wd,
                 name='Cell', **kwargs):
        """
        :param steps: 4
        :param multiplier: 4

        """
        super(Cell, self).__init__(name=name, **kwargs)

        self.wd = wd
        self.steps = steps
        self.multiplier = multiplier

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(ch, wd=wd, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(ch, k=1, s=1, wd=wd, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(ch, k=1, s=1, wd=wd, affine=False)

        self._ops = []
        for i in range(self.steps):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                strides = 2 if reduction and j < 2 else 1
                op = MixedOP(ch, strides=strides, wd=wd)
                self._ops.append(op)

    def call(self, s0, s1, weights, edge_weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for _ in range(self.steps):
            s = 0
            for j, h in enumerate(states):
                branch = self._ops[offset + j](h, weights[offset + j])
                s += edge_weights[offset + j] * branch
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self.multiplier:], axis=-1)


class SplitSoftmax(tf.keras.layers.Layer):
    """Split Softmax Layer"""
    def __init__(self, size_splits, name='SplitSoftmax', **kwargs):
        super(SplitSoftmax, self).__init__(name=name, **kwargs)
        self.size_splits = size_splits
        self.soft_max_func = Softmax(axis=-1)

    def call(self, value):
        return tf.concat(
            [self.soft_max_func(t) for t in tf.split(value, self.size_splits)],
            axis=0)


class SearchNetArch(object):
    """Search Network Architecture"""
    def __init__(self, cfg, steps=4, multiplier=4, stem_multiplier=3,
                 name='SearchModel'):
        self.cfg = cfg
        self.steps = steps
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.name = name

        self.arch_parameters = self._initialize_alphas()
        self.model = self._build_model()

    def _initialize_alphas(self):
        k = sum(range(2, 2 + self.steps))
        num_ops = len(PRIMITIVES)
        w_init = tf.random_normal_initializer()
        self.alphas_normal = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
            trainable=True, name='alphas_normal')
        self.alphas_reduce = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k, num_ops], dtype='float32'),
            trainable=True, name='alphas_reduce')
        self.betas_normal = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
            trainable=True, name='betas_normal')
        self.betas_reduce = tf.Variable(
            initial_value=1e-3 * w_init(shape=[k], dtype='float32'),
            trainable=True, name='betas_reduce')

        return [self.alphas_normal, self.alphas_reduce, self.betas_normal,
                self.betas_reduce]

    def _build_model(self):
        """Model"""
        logging.info(f"buliding {self.name}...")

        input_size = self.cfg['input_size'] # 32x32
        ch_init = self.cfg['init_channels'] # 16
        layers = self.cfg['layers'] # 8
        num_cls = self.cfg['num_classes'] # 10
        wd = self.cfg['weights_decay']

        # define model
        inputs = Input([input_size, input_size, 3], name='input_image')
        alphas_normal = Input([None], name='alphas_normal')
        alphas_reduce = Input([None], name='alphas_reduce')
        betas_normal = Input([], name='betas_normal')
        betas_reduce = Input([], name='betas_reduce')

        alphas_reduce_weights = Softmax(
            name='AlphasReduceSoftmax')(alphas_reduce)
        alphas_normal_weights = Softmax(
            name='AlphasNormalSoftmax')(alphas_normal)
        betas_reduce_weights = SplitSoftmax(
            range(2, 2 + self.steps), name='BetasReduceSoftmax')(betas_reduce)
        betas_normal_weights = SplitSoftmax(
            range(2, 2 + self.steps), name='BetasNormalSoftmax')(betas_normal)

        ch_curr = self.stem_multiplier * ch_init
        s0 = s1 = Sequential([
            Conv2D(filters=ch_curr, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=kernel_init(),
                   kernel_regularizer=regularizer(wd), use_bias=False),
            BatchNormalization(affine=True)], name='stem')(inputs)

        ch_curr = ch_init
        reduction_prev = False
        for layer_index in range(layers): # layers: 8
            # for layer in the middel [1/3, 2/3], reduce via stride=2
            if layer_index in [layers // 3, 2 * layers // 3]:
                ch_curr *= 2
                reduction = True
                weights = alphas_reduce_weights
                edge_weights = betas_reduce_weights
            else:
                reduction = False
                weights = alphas_normal_weights
                edge_weights = betas_normal_weights

            # the output channels = multiplier * ch_curr
            cell = Cell(self.steps, self.multiplier, ch_curr, reduction,
                        reduction_prev, wd, name=f'Cell_{layer_index}')

            s0, s1 = s1, cell(s0, s1, weights, edge_weights)

            # update reduction_prev
            reduction_prev = reduction

        fea = GlobalAveragePooling2D()(s1)

        logits = Dense(num_cls, kernel_initializer=kernel_init(),
                       kernel_regularizer=regularizer(wd))(Flatten()(fea))

        return Model(
            (inputs, alphas_normal, alphas_reduce, betas_normal, betas_reduce),
            logits, name=self.name)

    def get_genotype(self):
        """get genotype"""
        def _parse(weights, edge_weights):
            n = 2
            start = 0
            gene = []
            for i in range(self.steps): # for each node
                end = start + n
                w = weights[start:end].copy()
                ew = edge_weights[start:end].copy()

                # fused weights
                for j in range(n):
                    w[j, :] = w[j, :] * ew[j]

                # pick the top 2 edges (k = 2).
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(w[x][k] for k in range(len(w[x]))
                                       if k != PRIMITIVES.index('none'))
                    )[:2]

                # pick the top best op, and append into genotype.
                for j in edges:
                    k_best = None
                    for k in range(len(w[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or w[j][k] > w[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        gene_reduce = _parse(
            Softmax()(self.alphas_reduce).numpy(),
            SplitSoftmax(range(2, 2 + self.steps))(self.betas_reduce).numpy())
        gene_normal = _parse(
            Softmax()(self.alphas_normal).numpy(),
            SplitSoftmax(range(2, 2 + self.steps))(self.betas_normal).numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat,
                            reduce=gene_reduce, reduce_concat=concat)

        return genotype
