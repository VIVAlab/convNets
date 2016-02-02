require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

print('Model')
torch.setdefaulttensortype('torch.FloatTensor')

model = nn.Sequential()
if  (opt.load ~= "" ) then
    
model = torch.load('/home/jblan016/FaceDetection/48calibnet/results/model.net')

if opt.type == 'cuda' then 
   model = model:cuda()
end

    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 
else
----------------------
model:add(nn.SpatialConvolutionMM(1,64,5,5,1,1)) --O(64x44x44)-- #input-planes = 3, #output-planes=16, 

model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) --O(64x22x22) -- MaxPooling size=3, stride=2
model:add(nn.ReLU())


model:add(nn.SpatialConvolutionMM(64,64,5,5,1,1)) --O(64x18x18)

model:add(nn.ReLU())
-------------
model:add(nn.SpatialConvolutionMM(64,256,18,18,1,1)) 
model:add(nn.ReLU())
model:add(nn.Reshape(256))
model:add(nn.Linear(256,45))
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

