require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers




torch.setdefaulttensortype('torch.FloatTensor')
if  (opt.load ~= "") then
    model = torch.load(opt.load)
    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 	
else	
--with parallel path

model = nn.Sequential()
c = nn.ConcatTable()
P1 = nn.Sequential()
P1:add(nn.SpatialDropout())
P1:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))
P1:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
P1:add(nn.ReLU())
P1:add(nn.SpatialDropout())
P1:add(nn.SpatialConvolutionMM(64,128,10,10,1,1))
P1:add(nn.ReLU())
P1:add(nn.Reshape(128))


P2 = nn.Sequential()
-- comment because we are recycling weights
--[[
P2:add(nn.SpatialSubSampling(3,2,2,2,2))
--P2:add(nn.SpatialDropout())
P2:add(nn.SpatialConvolutionMM(3,16,3,3,1,1))
P2:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
P2:add(nn.ReLU())
--P2:add(nn.SpatialDropout())
--]]
P2:add(nn.SpatialConvolutionMM(16,16,5,5,1,1))
P2:add(nn.ReLU())
P2:add(nn.Reshape(16))


c:add(P1)
c:add(P2)
model:add(c)

model:add(nn.JoinTable(2))
model:add(nn.Dropout())
model:add(nn.Linear(128+16,2))
model:add(nn.LogSoftMax())

--without parallel path

--[[model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialConvolutionMM(64,128,9,9,1,1))
model:add(nn.ReLU())
model:add(nn.Reshape(128))

model:add(nn.Linear(128,2))
model:add(nn.LogSoftMax())--]]

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

