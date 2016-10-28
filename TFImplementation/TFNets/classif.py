from network import Network

class classif(Network):
    def setup(self):
        (self.feed('input')
             .flatten(name='classif.Flatten_8', OutputShape=32)
             .fc(2, relu=False, name='classif.InnerProduct_9')
             .softmax(name='classif.Softmax_10'))
