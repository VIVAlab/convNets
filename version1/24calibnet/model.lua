require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

print('Model')
torch.setdefaulttensortype('torch.FloatTensor')

model = nn.Sequential()
--Layer one --Convolutional layer
model:add(nn.SpatialConvolutionMM(3,32,5,5,1,1)) -- #input-planes = 3, #output-planes=16, 

--layer two --Max pooling layer
model:add(nn.SpatialMaxPooling(3,3,2,2)) -- MaxPooling size=3, stride=2
model:add(nn.ReLU())

--Layer three -FullyConnected Layer
model:add(nn.SpatialConvolutionMM(32,64,9,9,1,1))
model:add(nn.ReLU())

--Layer five --Labels
model:add(nn.Reshape(64))

model:add(nn.Linear(64,45))
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

