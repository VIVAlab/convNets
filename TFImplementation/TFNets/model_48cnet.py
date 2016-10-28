from network import Network

class model_48cnet(Network):
    def setup(self):
        (self.feed('input')
             .conv(5, 5, 64, 1, 1, padding='VALID', relu=False, name='model_48cnet.SpatialConvolution_0')
             .max_pool(3, 3, 2, 2, name='model_48cnet.Pooling_1')
             .relu(name='model_48cnet.ReLU_2')
             .conv(5, 5, 64, 1, 1, padding='VALID', relu=False, name='model_48cnet.SpatialConvolution_3')
             .conv(18, 18, 256, 1, 1, padding='VALID', name='model_48cnet.SpatialConvolution_4')
             .flatten(name='model_48cnet.Flatten_6', OutputShape=256)
             .fc(45, relu=False, name='model_48cnet.InnerProduct_7')
             .softmax(name='model_48cnet.Softmax_8'))
