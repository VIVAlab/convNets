require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'inn'
require 'cudnn'


torch.setdefaulttensortype('torch.FloatTensor')
--[[input = nn.Identity()()
P1 = nn.Reshape(9*9*64,false)(nn.ReLU()((nn.SpatialMaxPooling(3,3,2,2)(nn.SpatialConvolutionMM(3,64,5,5,1,1)(input)))))
P2 = nn.Reshape(4*4*16,false)(nn.ReLU()((nn.SpatialMaxPooling(3,3,2,2)(nn.SpatialConvolutionMM(3,16,3,3,1,1)(nn.SpatialMaxPooling(2,2,2,2)(input))))))

L = nn.LogSoftMax()(nn.Linear(128,2)(nn.ReLU()(nn.Linear(81*64+16*16,128)(nn.JoinTable(1)({P1,P2})))))

model = nn.gModule({input},{L})

input = nn.Identity()()
P1 = nn.LogSoftMax()(nn.Linear(81*64,128)(nn.ReLU()(((nn.ReLU()((nn.SpatialMaxPooling(3,3,2,2)(nn.SpatialConvolutionMM(3,64,5,5,1,1)(input)))))))))

model = nn.gModule({input},{P1})
--]]
if  (opt.load ~= "" ) then
    model = torch.load(opt.load)
    print(sys.COLORS.blue .. '**Pre-trained model loaded**') 	
else

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))    --O(64x60x60)
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))        --O(64x30x30)
model:add(nn.ReLU())
--P3:add(inn.SpatialCrossResponseNormalization(9))
--model:add(nn.SpatialDropout())
model:add(nn.SpatialConvolutionMM(64,64,5,5,1,1))   --O(64x26x26)   
--:add(inn.SpatialCrossResponseNormalization(9))
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))        --O(64x13x13)
model:add(nn.ReLU())
--model:add(nn.SpatialDropout()) 
model:add(nn.SpatialConvolutionMM(64,256,13,13,1,1))  --
model:add(nn.ReLU())
model:add(nn.Reshape(256))
model:add(nn.Dropout(.25))
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

