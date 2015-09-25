require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers




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
c = nn.ConcatTable()
P1 = nn.Sequential()
P1:add(nn.SpatialSubSampling(3,2,2,2,2))
P1:add(nn.SpatialDropout())
P1:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))
P1:add(nn.SpatialMaxPooling(3,3,2,2))
P1:add(nn.ReLU())
P1:add(nn.Dropout())
P1:add(nn.SpatialConvolutionMM(64,128,9,9,1,1))
P1:add(nn.ReLU())
P1:add(nn.Reshape(128))


P2 = nn.Sequential()
P2:add(nn.SpatialSubSampling(3,4,4,4,4))
P2:add(nn.SpatialDropout())
P2:add(nn.SpatialConvolutionMM(3,16,3,3,1,1))
P2:add(nn.SpatialMaxPooling(3,3,2,2))
P2:add(nn.ReLU())
P2:add(nn.Dropout())
P2:add(nn.SpatialConvolutionMM(16,16,4,4,1,1))
P2:add(nn.ReLU())
P2:add(nn.Reshape(16))


ker = torch.ones(9)
P3 = nn.Sequential()
P3:add(nn.SpatialDropout())
P3:add(nn.SpatialConvolutionMM(3,64,5,5,1,1))
P3:add(nn.SpatialMaxPooling(3,3,2,2))
P3:add(nn.ReLU())
--P3:add(nn.SpatialSubtractiveNormalization(64,ker))
P3:add(nn.SpatialDropout())
P3:add(nn.SpatialConvolutionMM(64,64,5,5,1,1))
--P3:add(nn.SpatialSubtractiveNormalization(64,ker))
P3:add(nn.SpatialMaxPooling(3,3,2,2))
P3:add(nn.ReLU())
P3:add(nn.Dropout())
P3:add(nn.SpatialConvolutionMM(64,256,8,8,1,1))
P3:add(nn.ReLU())
P3:add(nn.Reshape(256))


c:add(P1)
c:add(P2)
c:add(P3)
model:add(c)

model:add(nn.JoinTable(2))
model:add(nn.Dropout())
model:add(nn.Linear(256+128+16,2))
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

