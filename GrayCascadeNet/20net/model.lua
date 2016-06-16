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

model:add(nn.SpatialConvolutionMM(1,32,3,3,1,1)) --I(1x20x20)->O(32x18x18)
--model:add(nn.Dropout())
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --O(32x9x9)
model:add(nn.ReLU())
model:add(nn.SpatialConvolutionMM(32,32,3,3,1,1))--O(32x7x7)
--model:add(nn.Dropout())
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2,0,0)) --O(32x3x3)
model:add(nn.SpatialConvolutionMM(32,32,3,3,1,1))--O(32x1x1)
model:add(nn.ReLU())
model:add(nn.Reshape(32))
model:add(nn.Linear(32,2))
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

