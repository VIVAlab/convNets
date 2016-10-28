from network import Network

class model_20net(Network):
    def setup(self):
        (self.feed('input')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='model_20net.SpatialConvolution_0')
             .max_pool(3, 3, 2, 2, name='model_20net.Pooling_1')
             .relu(name='model_20net.ReLU_2')
             .conv(3, 3, 32, 1, 1, padding='VALID', name='model_20net.SpatialConvolution_3')
             .max_pool(3, 3, 2, 2, padding='VALID', name='model_20net.Pooling_5')
             .conv(3, 3, 32, 1, 1, padding='VALID', name='model_20net.SpatialConvolution_6')
             .flatten(name='model_20net.Flatten_8', OutputShape=32)
             .fc(2, relu=False, name='model_20net.InnerProduct_9')
             .softmax(name='model_20net.Softmax_10'))
