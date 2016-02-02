require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

print('Model')
torch.setdefaulttensortype('torch.FloatTensor')

if  (opt.load ~= "") then
    network1 = torch.load(opt.load)
--[[
model = nn.Sequential()
model:add(network1.modules[1])
model:add(nn.Dropout())
	for i=2,4 do
		model:add(network1.modules[i])
	end
model:add(nn.Dropout())
	for i=5,7 do
		model:add(network1.modules[i])
	end
model:add(nn.Dropout())
model:add(network1.modules[8])
	if opt.type=='cuda' then
		model:cuda()
	end
network1=nil

--]]
    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 	


else	
model = nn.Sequential()
--Layers 1-3 are not loaded from 12net model (see train.lua)--------------------
model:add(nn.SpatialConvolutionMM(1,16,3,3,1,1)) --I(1x12x12)->O(16x10x10)--
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --O(16x5x5)                  --
model:add(nn.ReLU())							  --
--------------------------------------------------------------------------
model:add(nn.SpatialConvolutionMM(16,128,5,5,1,1)) 
model:add(nn.ReLU())

--Layer 4: labels
model:add(nn.Reshape(128))
model:add(nn.Linear(128,45))
model:add(nn.LogSoftMax())


end
-- Loss: NLL
loss = nn.ClassNLLCriterion()
if opt.type=='cuda' then
		model:cuda()
end

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

