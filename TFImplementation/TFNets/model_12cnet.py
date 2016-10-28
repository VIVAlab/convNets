from network import Network

class model_12cnet(Network):
    def setup(self):
        (self.feed('input')
             .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='model_12cnet.SpatialConvolution_0')
             .max_pool(3, 3, 2, 2, name='model_12cnet.Pooling_1')
             .relu(name='model_12cnet.ReLU_2')
             .conv(5, 5, 128, 1, 1, padding='VALID', name='model_12cnet.SpatialConvolution_3')
             .flatten(name='model_12cnet.Flatten_5', OutputShape=128)
             .fc(45, relu=False, name='model_12cnet.InnerProduct_6')
             .softmax(name='model_12cnet.Softmax_7'))
