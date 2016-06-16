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

model:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))    --O(64x44x44)
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))        --O(64x22x22)
model:add(nn.ReLU())
--P3:add(inn.SpatialCrossResponseNormalization(9))
--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(64,64,5,5,1,1))   --O(64x18x18)   
--:add(inn.SpatialCrossResponseNormalization(9))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))        --O(64x9x9)
model:add(nn.ReLU())
--model:add(nn.SpatialDropout()) 
model:add(nn.SpatialConvolutionMM(64,256,9,9,1,1))  --
model:add(nn.ReLU())
model:add(nn.Reshape(256))
model:add(nn.Dropout(.5))
model:add(nn.Linear(256,256))
model:add(nn.ReLU())
model:add(nn.Dropout(.5))
model:add(nn.Linear(256,2))
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

