require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

print('Model')
torch.setdefaulttensortype('torch.FloatTensor')

if  (opt.load ~= "") then
    model = torch.load(opt.load)
    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 	
else	
model = nn.Sequential()
--Layers 1-2 are loaded from 12net model (see train.lua)
--TODO: to recycle, comment the 3 following lines:
model:add(nn.SpatialConvolutionMM(3,16,3,3,1,1)) -- I(3x12x12)->O(16x10x10)
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) -- I(16x10x10)->O(16x5x5)
model:add(nn.ReLU())


--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(16,128,5,5,1,1)) -- I(16x5x5)->O(128x1x1)
model:add(nn.ReLU())

--Layer 4: labels
model:add(nn.Reshape(128))
--model:add(nn.SpatialDropout())
model:add(nn.Linear(128,45))
model:add(nn.LogSoftMax())


end
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

