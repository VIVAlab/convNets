# TensorFlow Implementation
Used torch2caffe and caffe-tensorflow, with some changes outlined below, to convert the individual networks.
demo_TF.py contains the cascade in python, and using TensorFlow rather than Lua/Torch.
TFNets contains the models, weights, and mean/standard deviation changes for the images.



# Steps for Torch2tensorflow:
Torch2caffe (see command below)

	- Commented out line 87 of caffe_layers.py, which adds the -1 to the # of axis for caffe.InnerProduct.
	- Commented out lines 166 and 167 of caffe_layers.py.
	- Change line 47 of torch_layers.lua from typename="caffe.LogSoftmax" to typename="caffe.Softmax".
	- For 48net and 48cnet, need to add "torch_net:float()" to the beginning of the M.convert function in lib.lua
	- For verification code: Changed logging.infof() to print() in lines 55 and 141 of lib.lua. (Verification probably fails because of the change we made to account for the lack of logsoftmax layer in caffe.)
Update .prototxt and .caffemodel (see command below)
- Add to .prototxt:
	- first line: name: "SomeName" (SomeName can be any name)(or add it to the python TF code after caffe2TF, by adding SomeName before (Network))
	- Remove any dropout layers (should be in-place operations) (Caffe2tensorflow cannot convert dropout layer? Dropout layer is only used in training though.)
- Add flatten layer conversion (mostly based on https://github.com/ethereon/caffe-tensorflow/issues/34):
	- In layers.py, change name associated with 'Flatten' (to shape_flatten)
	- In shapes.py, added: def shape_flatten(node)
	- In transormer.py, need to add map_flatten class to class TensorFlowMapper(NodeMapper)
	- In network.py add:
			@layer
				def flatten(self,input,name, OutputShape):
					return tf.reshape(input,[-1, OutputShape])
      And add ", extrastride=1" to the end of line 36.
	- This requires that we comment out the line "axis=-1" in .prototxt, for layer InnerProduct (or replace -1 with 1)??
Caffe-tensorflow (see command below)
- Replace "from kaffe.tensorflow import Network" with "from network import Network" for each network file
- Can easily implement Network definition from kaffe.tensorflow (in caffe-tensorflow), as it has no dependencies on the rest of caffe-tensorflow.


For mean and standard deviation file conversion, run the following commands in torch:
	temp = torch.load('/home/danlaptop/FaceDetection/Cascade/GrayCascadeNet/CascadeWeights/mean_12cnet.dat')
	torch.save('/home/danlaptop/FaceDetection/Cascade/GrayCascadeNet/CascadeWeights_Test/mean_12cnet.txt', temp[1], 'ascii')
	temp (this line is only to let us compare the .txt output with the data loaded from .dat files)


Torch->Caffe command:
	th torch2caffe/torch2caffe.lua --input /home/danlaptop/fb-caffe-exts/CascadeWeights/model_20net.net --prototxt /tmp/torch2caffe/model_20net.prototxt --caffemodel /tmp/torch2caffe/model_20net.caffemodel 1 1 19 19
	
	(Use 1 1 11 11 for 12cnet, 1 1 47 47 for 48cnet and 1 1 45 45 48cnet) (Can't use 12, 20 and 48 because the pooling layers are not implemented in the same way in torch and caffe. The number is not used in TF, so this should not matter)

Will need to update to latest .prototxt version (run these commands from the /caffe/build/tools folder):
	For binary: ./upgrade_net_proto_binary /tmp/torch2caffe/model_20net.caffemodel /tmp/caffe_update/model_20net.caffemodel
	For model: ./upgrade_net_proto_text /tmp/torch2caffe/model_20net.prototxt /tmp/caffe_update/model_20net.prototxt
  
Caffe->TF command (with weights) (run this command from the caffe-tensorflow folder):
	./convert.py /tmp/caffe_update/model_20net.prototxt --caffemodel /tmp/caffe_update/model_20net.caffemodel --data-output-path /tmp/caffe2tensorflow/model_20net_data.npy --code-output-path /tmp/caffe2tensorflow/model_20net_code.py
