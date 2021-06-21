from net_helper import *

class Model3d():
    
    def __init__(self, input_x, **kwargs):
        self.input = input_x
        self.depth = kwargs['depth']
        self.layer_channel_list = kwargs['channels']
        self.groups = kwargs['groups']
        self.in_shape = kwargs['input_size']
        self.n_class = kwargs['n_class']
        self.activation = kwargs['activation'] # last layer
        self.init_method = kwargs.get('init_method', 'default')
        self.keep_prob = kwargs.get('keep_prob',1.0)
        self.normalize = kwargs.get('normalize', 'bn')
        self.pool_strides = kwargs.get('pool_strides', (2,2,2))
        self.batch_size = kwargs.get('batch_size', 1)
        self.training = kwargs.get('training', True)
        
        self.output = None
        
    def preduce_input(self):
        c = 1
        b = self.batch_size
        if len(self.in_shape) == 3:
            d, h, w = self.in_shape
        if len(self.in_shape) == 4:
            if self.in_shape[0] != self.batch_size:
                d, h, w, c = self.in_shape
            else:
                b, d, h, w  = self.in_shape
        in_tensor_shape = (b, d, h, w, c)
        self.input = tf.reshape(self.input, in_tensor_shape)
        
    def conv3d(self, input_x, kernel_size, out_channels):
        print('input.shape', input_x.shape, flush=True)
        cin = input_x.get_shape().as_list()[-1]
        initializer = get_initializer(self.init_method, kernel_size, cin)
        if initializer is None:
            # use default method
            output = tf.layers.conv3d(input_x, filters=out_channels, kernel_size=kernel_size, strides=1, padding='same') 
        else:
            output = tf.layers.conv3d(input_x, filters=out_channels, kernel_size=kernel_size, strides=1, kernel_initializer=initializer, padding='same') 
        # normalization
        if self.normalize == 'gn':
#             output = Group_norm3d(output, min(out_channels//2, 32))
            output = tf.contrib.layers.group_norm(inputs=output,groups=min(out_channels//2,32))
#             output = tf.contrib.layers.group_norm(inputs=output,groups=self.groups)
        else:
            output = tf.layers.batch_normalization(output, training=self.training)
        # relu activation
        output = tf.nn.relu(output)
        return output
    
    def conv3d_dilated(self, input_x, out_channels, drate=1):
        return tf.layers.conv3d(input_x, filters=out_channels, kernel_size=3, strides=1, dilation_rate=(drate,drate,1), padding='same')
    
    def max_pooling(self, input_x):
        return tf.layers.max_pooling3d(input_x, pool_size=(2, 2, 2), strides=self.pool_strides, padding='same')

    def conv_transpose(self, input_x, kernel_size, out_channels):
        cin = input_x.get_shape().as_list()[-1]
        initializer = get_initializer(self.init_method, kernel_size, cin)
        if initializer is None:
            output = tf.layers.conv3d_transpose(input_x, filters=out_channels, kernel_size=kernel_size, strides=self.pool_strides, padding='same')
        else:
            output = tf.layers.conv3d_transpose(input_x, filters=out_channels, kernel_size=kernel_size, strides=self.pool_strides, kernel_initializer=initializer, padding='same')
        return output
    
    def conv_block(self, input_x, layer_idx, num_layers, num_channels):
        print(f'conv{layer_idx}', flush=True)
        with tf.variable_scope(f'conv{layer_idx}',reuse=tf.AUTO_REUSE):
            conv1 = self.conv3d(input_x, 1, num_channels)
            x = input_x
            for i in range(num_layers):
                x = self.conv3d(x, 3, num_channels)
            if num_layers == 1:
                return x
            return conv1 + x
        
    def _up(self, input, simi_level_in, layer_idx, num_layers, num_channels, last_output=None):
        print(f'conv{layer_idx}', flush=True)
        with tf.variable_scope(f'conv{layer_idx}', reuse=tf.AUTO_REUSE):
            m = self.conv_transpose(input, 2, num_channels)
            n = tf.concat(values=[simi_level_in, m], axis=4)
            output = self.conv_block(n, layer_idx, num_layers, num_channels)
            if last_output != None:
                ch = last_output.get_shape().as_list()[-1]
        #       print('last data shape: ', last_output.shape)
                last_output = self.conv_transpose(last_output, 2, ch//2)
                copy_output = tf.concat([last_output, output], axis=4)
            else:
                copy_output = output
            return output, copy_output

    def level_block(self, m, layer_idx):
        # input: m
        last_data = None
        num_layers, out_ch = self.layer_channel_list[layer_idx+1]
        if layer_idx < self.depth:
            n = self.conv_block(m, layer_idx, num_layers, out_ch)
            m = self.max_pooling(n)
            m, last_data = self.level_block(m, layer_idx+1)
            m, last_data = self._up(m, n, 2*self.depth-layer_idx, num_layers, out_ch, last_data)
        else:
            # dropout
            if self.keep_prob != 1:
                m = tf.nn.dropout(m, self.keep_prob)
            m = self.conv_block(m, layer_idx, num_layers, out_ch)
        return m, last_data

    def head(self, input_, initializer):
        with tf.variable_scope(f'last_conv', reuse=tf.AUTO_REUSE):
            if initializer is None:
                output = tf.layers.conv3d(input_, filters=self.n_class, kernel_size=1, strides=1,
                                          padding='same')
            else:
                if self.activation == 'sigmoid':
                    nc = 1
                output = tf.layers.conv3d(input_, filters=nc, kernel_size=1, strides=1, padding='same',
                                          kernel_initializer=initializer)
            print('output.shape', output.shape, flush=True)
            if self.activation == 'sigmoid':
                self.output = tf.sigmoid(output)
            else:
                self.output = tf.nn.softmax(output)
            return output

    def inference(self):
        self.preduce_input()
        self.input = tf.identity(self.input, name='input')
        output, merge_deep_super = self.level_block(self.input, 0)
#         print('last output.shape', merge_deep_super.shape, flush=True)
        cin = merge_deep_super.get_shape().as_list()[-1]
        initializer = get_initializer(self.init_method, 1, cin)
        nc = self.n_class
        output = self.head(merge_deep_super, initializer)
        return self.output