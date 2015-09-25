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
--Layer one
--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(3,16,3,3,1,1)) -- #input-planes = 3, #output-planes=16, filter-size=6, stride=2
model:add(nn.SpatialMaxPooling(3,3,2,2)) -- MaxPooling size=3, stride=2
model:add(nn.ReLU())

--Layer three
--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(16,16,4,4,1,1))
model:add(nn.ReLU())

--Layer five
model:add(nn.Reshape(16))

--Layer six
--model:add(nn.Dropout())
model:add(nn.Linear(16,2))

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

