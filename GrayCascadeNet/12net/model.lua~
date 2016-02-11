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

model:add(nn.SpatialConvolutionMM(1,16,3,3,1,1)) --I(1x12x12)->O(16x10x10)
model:add(nn.Dropout())
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --O(16x5x5)
model:add(nn.ReLU())
model:add(nn.SpatialConvolutionMM(16,16,5,5,1,1))
model:add(nn.Dropout())
model:add(nn.ReLU())

model:add(nn.Reshape(16))

model:add(nn.Linear(16,2))

model:add(nn.Dropout())
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

