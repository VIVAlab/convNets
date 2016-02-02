require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers




torch.setdefaulttensortype('torch.FloatTensor')
if  (opt.load ~= "" ) then
    model1 = torch.load(opt.load)
    model =nn.Sequential()
	for i=1,6 do
	model:add(model1.modules[i])
	end
	for i=8,9 do
	model:add(model1.modules[i])
	end
    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 	
else


model = nn.Sequential()
--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(1,64,5,5,1,1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
--model:add(nn.SpatialDropout(.5))
model:add(nn.SpatialConvolutionMM(64,128,10,10,1,1))
model:add(nn.ReLU())
model:add(nn.Reshape(128))
--model:add(nn.Dropout(.25))
model:add(nn.Linear(128,2))
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

