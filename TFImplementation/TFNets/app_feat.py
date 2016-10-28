from network import Network

class app_feat(Network):
    def setup(self):
        (self.feed('input')
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='app_feat.SpatialConvolution_0')
             .max_pool(3, 3, 2, 2, name='app_feat.Pooling_1')
             .relu(name='app_feat.ReLU_2')
             .conv(3, 3, 32, 1, 1, padding='VALID', name='app_feat.SpatialConvolution_3')
             .max_pool(3, 3, 2, 2, padding='VALID', name='app_feat.Pooling_5')
             .conv(3, 3, 32, 1, 1, padding='VALID', name='app_feat.SpatialConvolution_6'))
             #.flatten(name='app_feat.Flatten_8', OutputShape=32)
             #.fc(2, relu=False, name='app_feat.InnerProduct_9')
             #.softmax(name='app_feat.Softmax_10'))
