require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

print('Model')
torch.setdefaulttensortype('torch.FloatTensor')

model = nn.Sequential()
--Layer 1: Convolutional Layer
model:add(nn.SpatialConvolutionMM(3,64,5,5,1,1)) -- #input-planes = 3, #output-planes=16, 
--Layer 2: Max-pooling layer +RELU
model:add(nn.SpatialMaxPooling(3,3,2,2)) -- MaxPooling size=3, stride=2
model:add(nn.ReLU())
--Layer 3: TODO INSERT NORMALISATION LAYER
--Layer 4: Convolutional layer
model:add(nn.SpatialConvolutionMM(64,64,5,5,1,1))
--Layer 5: TODO INSERT NORMALISATION LAYER
--Layer 6: Fully connected layer +RELU

model:add(nn.SpatialConvolutionMM(64,256,17,17,1,1))
model:add(nn.ReLU())

--layer 7: labels
model:add(nn.Reshape(256))
model:add(nn.Linear(256,45))
model:add(nn.LogSoftMax())


-- Loss: NLL
loss = nn.ClassNLLCriterion()


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> here is the CNN:')
print(model)

if opt.type == 'cuda' then
   model:cuda()
   loss:cuda()
end

-- return package:
return {
   model = model,
   loss = loss,
}
