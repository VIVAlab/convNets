from network import Network

class model_48net(Network):
    def setup(self):
        (self.feed('input')
             .conv(5, 5, 64, 1, 1, padding='VALID', relu=False, name='model_48net.SpatialConvolution_0')
             .max_pool(3, 3, 2, 2, name='model_48net.Pooling_1')
             .relu(name='model_48net.ReLU_2')
             .conv(5, 5, 64, 1, 1, padding='VALID', name='model_48net.SpatialConvolution_3')
             .max_pool(3, 3, 2, 2, name='model_48net.Pooling_5')
             .relu(name='model_48net.ReLU_6')
             .conv(9, 9, 128, 1, 1, padding='VALID', name='model_48net.SpatialConvolution_7')
             .flatten(name='model_48net.Flatten_10', OutputShape=128)
             .fc(2, relu=False, name='model_48net.InnerProduct_11')
             .softmax(name='model_48net.Softmax_12'))
