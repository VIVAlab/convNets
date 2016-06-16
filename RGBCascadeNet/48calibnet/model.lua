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
--Layer 1: Convolutional Layer
model:add(nn.SpatialConvolutionMM(3,64,5,5,1,1)) -- I(3x48x48)->O(64x44x44)
--Layer 2: Max-pooling layer +RELU
model:add(nn.SpatialMaxPooling(3,3,2,2,1,1)) -- I(64x44x44)->O(64x22x22)
model:add(nn.ReLU())
--Layer 3: TODO INSERT NORMALISATION LAYER
--Layer 4: Convolutional layer
model:add(nn.SpatialConvolutionMM(64,64,5,5,1,1))-- I(64x22x22)->O(64x18x18)
--Layer 5: TODO INSERT NORMALISATION LAYER
--Layer 6: Fully connected layer +RELU

model:add(nn.SpatialConvolutionMM(64,256,18,18,1,1))  --I(256x18x18)->O(256x1x1)
model:add(nn.ReLU())

--layer 7: labels
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

