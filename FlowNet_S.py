#Use tensorflow 2.3 or newer
from submodules import *


class FlowNet_S(tf.keras.Model):

    def __init__(self, batchNorm=False):
        super().__init__()
        self.batchNorm = batchNorm
        # Just the normal everyday good ol' 2D convolution layers
        self.conv1 = conv2D(self.batchNorm, filters=64, kernel_size=7, stride=2, name='conv1')
        self.conv2 = conv2D(self.batchNorm, filters=128, kernel_size=5, stride=2, name='conv2')
        self.conv3 = conv2D(self.batchNorm, filters=256, kernel_size=5, stride=2, name='conv3')
        self.conv3_1 = conv2D(self.batchNorm, filters=256, name='conv3_1')
        self.conv4 = conv2D(self.batchNorm, filters=512, stride=2, name='conv4')
        self.conv4_1 = conv2D(self.batchNorm, filters=512, name='conv4_1')
        self.conv5 = conv2D(self.batchNorm, filters=512, stride=2, name='conv5')
        self.conv5_1 = conv2D(self.batchNorm, filters=512, name='conv5_1')
        self.conv6 = conv2D(self.batchNorm, filters=1024, stride=2, name='conv6')
        self.conv6_1 = conv2D(self.batchNorm, filters=1024, name='conv6_1')

        # These layers are 2D transposed convolutions. Its not exactly the reverse of a convolution, however, doing this
        # type of convolution will increase the output feature space, i.e., increase the height and width of the
        # output w.r.t the input, which is the opposite of what a regular 2D convolution does.
        self.deconv5 = deconv(512, name='deconv5')
        self.deconv4 = deconv(256, name='deconv4')
        self.deconv3 = deconv(128, name='deconv3')
        self.deconv2 = deconv(64, name='deconv2')

        # These layers are the usual 2D convolutions but with no activation applied to the convolution. It represents
        # the intermediate optical flow with two channels.
        self.predict_flow6 = predict_flow(name='predict_flow6')
        self.predict_flow5 = predict_flow(name='predict_flow5')
        self.predict_flow4 = predict_flow(name='predict_flow4')
        self.predict_flow3 = predict_flow(name='predict_flow3')
        self.predict_flow2 = predict_flow(name='predict_flow2')

        # These layers are the 2D transposed convolutions but with no activation applied to them. It respresents
        # upscaling of the intermediate optical flow fields.
        self.upsampled_flow6_to_5 = deconv(2, bias=False, activation=False, name='upsampled_flow6_to_5')
        self.upsampled_flow5_to_4 = deconv(2, bias=False, activation=False, name='upsampled_flow5_to_4')
        self.upsampled_flow4_to_3 = deconv(2, bias=False, activation=False, name='upsampled_flow4_to_3')
        self.upsampled_flow3_to_2 = deconv(2, bias=False, activation=False, name='upsampled_flow3_to_2')


    def call(self, inputs, training=None):
        # Defining the forward pass
        out_conv1 = self.conv1(inputs)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(antipad(self.upsampled_flow6_to_5(flow6)), out_conv5)
        out_deconv5 = crop_like(antipad(self.deconv5(out_conv6)), out_conv5)

        concat5 = tf.keras.layers.concatenate([out_conv5, out_deconv5, flow6_up], axis=-1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(antipad(self.upsampled_flow5_to_4(flow5)), out_conv4)
        out_deconv4 = crop_like(antipad(self.deconv4(concat5)), out_conv4)

        concat4 = tf.keras.layers.concatenate([out_conv4, out_deconv4, flow5_up], axis=-1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(antipad(self.upsampled_flow4_to_3(flow4)), out_conv3)
        out_deconv3 = crop_like(antipad(self.deconv3(concat4)), out_conv3)

        concat3 = tf.keras.layers.concatenate([out_conv3, out_deconv3, flow4_up], axis=-1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(antipad(self.upsampled_flow3_to_2(flow3)), out_conv2)
        out_deconv2 = crop_like(antipad(self.deconv2(concat3)), out_conv2)

        concat2 = tf.keras.layers.concatenate([out_conv2, out_deconv2, flow3_up], axis=-1)
        flow2 = self.predict_flow2(concat2)

        if training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2
