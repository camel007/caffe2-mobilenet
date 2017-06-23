## @package moiblenet
# Module caffe2.python.models.mobilenet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import brew

'''
Utility for creating MobileNets
See "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" by Andrew G. Howard et. al. 2017
'''


class MobileNetBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, model, prev_blob, no_bias, is_test, spatial_bn_mom=0.9):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0

    def add_conv(self, in_filters, out_filters, kernel, stride=1, pad=0):
        self.comp_idx += 1
        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_group_conv(self, in_filters, out_filters, kernel, stride=1, pad=0, group=1):
        self.comp_idx += 1
        self.prev_blob = brew.group_conv(
            self.model,
            self.prev_blob,
            'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
            group=group,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            'comp_%d_spatbn_%d' % (self.comp_count, self.comp_idx),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    def add_simple_block(
            self,
            input_filters,
            output_filters,
            down_sampling=False,
            spatial_batch_norm=True
    ):
        self.comp_idx = 0

        # 3x3
        self.add_group_conv(
            in_filters=input_filters,
            out_filters=input_filters,
            kernel=3,
            stride=(1 if down_sampling is False else 2),
            pad=1,
            group=input_filters
        )
        if spatial_batch_norm:
            self.add_spatial_bn(input_filters)
        self.add_relu()

        self.add_conv(
            input_filters,
            output_filters,
            kernel=1,
            stride=1,
            pad=0
        )
        if spatial_batch_norm:
            self.add_spatial_bn(output_filters)
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1


def create_mobilenet(
        model, data, num_input_channels, num_labels, label, is_test=False
):
    '''
    Create residual net for smaller images (sec 4.2 of He et. al (2015))
    num_groups = 'n' in the paper
    '''
    # conv1
    brew.conv(
        model,
        data,
        'conv1',
        3,
        32,
        weight_init=("MSRAFill", {}),
        kernel=3,
        stride=1,
        pad=1,
        no_bias=True,
    )
    brew.spatial_bn(
        model, 'conv1', 'conv1_spatbn', 32, epsilon=1e-3, is_test=is_test
    )
    brew.relu(model, 'conv1_spatbn', 'relu1')

    builder = MobileNetBuilder(model, 'relu1', no_bias=True, is_test=is_test)

    # block1
    builder.add_simple_block(input_filters=32, output_filters=64, down_sampling=False, spatial_batch_norm=True)
    # block2
    builder.add_simple_block(input_filters=64, output_filters=128, down_sampling=True, spatial_batch_norm=True)
    # block3
    builder.add_simple_block(input_filters=128, output_filters=128, down_sampling=False, spatial_batch_norm=True)
    # block4
    builder.add_simple_block(input_filters=128, output_filters=256, down_sampling=False, spatial_batch_norm=True)
    # block5
    builder.add_simple_block(input_filters=256, output_filters=256, down_sampling=False, spatial_batch_norm=True)
    # block6
    builder.add_simple_block(input_filters=256, output_filters=512, down_sampling=True, spatial_batch_norm=True)
    # block7-11
    for i in xrange(7, 12):
        builder.add_simple_block(input_filters=512, output_filters=512, down_sampling=False, spatial_batch_norm=True)
    # block12
    builder.add_simple_block(input_filters=512, output_filters=1024, down_sampling=True, spatial_batch_norm=True)
    # block13
    builder.add_simple_block(input_filters=1024, output_filters=1024, down_sampling=False, spatial_batch_norm=True)

    # Final layers
    brew.average_pool(
        model, builder.prev_blob, 'final_avg', kernel=4, stride=1)
    last_out = brew.fc(model, 'final_avg', 'last_out', 1024, num_labels)

    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ['softmax', 'loss'],
        )

        return (softmax, loss)
    else:
        return brew.softmax(model, 'last_out', 'softmax')
