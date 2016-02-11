----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to SGD.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'Flip'
print('Train')
----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss


----------------------------------------------------------------------
-- Save light network tools:
function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor() end
   module.gradWeight = nil
   module.output     = torch.Tensor()
   module.fgradInput = nil
   module.gradInput  = nil
end

function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining some tools')

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState 
local optimMethod
if opt.optimization=='sgd' then
optimState= {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}
optimMethod=optim.sgd
elseif opt.optimization=='cg' then
optimState={
    verbose=true,
    maxIter=opt.maxIter
}
optimMethod=optim.cg
elseif opt.optimization=='lbfgs' then
optimState = {
   lineSearch = optim.lswolfe,
   maxIter = opt.maxIter,  --epochs
   verbose = true
}
optimMethod=optim.lbfgs
else
error('unknown optimizer')
end
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local batchPOS=math.ceil(opt.batchSize/(opt.trainingratio+1))
local batchNEG=opt.batchSize-batchPOS
local batchNoPOS=math.floor(trainDataPOS.data:size(1)/batchPOS)
local batchNoNEG=math.floor(trainDataNEG.data:size(1)/batchNEG)
print('b#-='..batchNoNEG)
print('b#+='..batchNoPOS)
local xNEG = torch.Tensor(batchNEG,trainDataNEG.data:size(2), 
         trainDataNEG.data:size(3), 2*trainDataNEG.data:size(4)) 
local xPOS = torch.Tensor(batchPOS,trainDataPOS.data:size(2), 
         trainDataPOS.data:size(3), 2*trainDataPOS.data:size(4))   
local x = torch.Tensor(opt.batchSize,trainDataPOS.data:size(2), 
         trainDataPOS.data:size(3), 2*trainDataPOS.data:size(4))   

local ytNEG = torch.Tensor(batchNEG)
local ytPOS = torch.Tensor(batchPOS)
local yt = torch.Tensor(opt.batchSize)
-- set to float
   xNEG=xNEG:float()
   xPOS=xPOS:float()
   x = x:float()
   ytPOS=ytPOS:float()
   ytNEG=ytNEG:float()
   yt = yt:float()

local shufflePOS = torch.Tensor(trainDataPOS:size(1)):fill(0)
local shuffleNEG = torch.Tensor(trainDataNEG:size(1)):fill(0)
if opt.type == 'cuda' then 
   xNEG=xNEG:cuda()
   xPOS=xPOS:cuda()
   x = x:cuda()

   ytPOS=ytPOS:cuda()
   ytNEG=ytNEG:cuda()
   yt = yt:cuda()
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epochPOS
local epochNEG
local shufflePOS_FLAG=true
local shuffleNEG_FLAG=true
-- data index counters
local tNEG=0 
local tPOS=0
local dimension=4 --x dim
local k=torch.range(12,1,-1):type('torch.LongTensor') --should be 12,11,10,...,1
local function train(trainDataPOS,trainDataNEG)   ----------------------=================================TRAIN============

   -- epoch tracker
   epochPOS = epochPOS or 1
   epochNEG = epochNEG or 1
   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
	if shufflePOS_FLAG==true then
    shufflePOS = torch.randperm(trainDataPOS:size(1))
          shufflePOS_FLAG=false
	end
	if shuffleNEG_FLAG==true then
    shuffleNEG = torch.randperm(trainDataNEG:size(1))
          shuffleNEG_FLAG=false
	end
   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on positive training data:') 
   print("==> online positive data epoch # " .. epochPOS .. ' [batchSize(+) = ' .. batchPOS .. '].'..' #data(+)='..trainDataPOS.data:size(1)) 
   print("==> online negative data epoch # " .. epochNEG .. ' [batchSize(-) = ' .. batchNEG .. '].'..' #data(-)='..trainDataNEG.data:size(1))

   while true do  -- batch loop --------------------------------------
      
      -- disp progress
      --xlua.progress(t, trainData:size())
      collectgarbage()
      -- positive data batch fits?
      tNEG=tNEG+1
	--print('t-='..tNEG)
      if (tNEG) > batchNoNEG then
         tNEG=0
         shuffleNEGflag=true
         epochNEG=epochNEG+1
         break
      end
      -- negative data batch fits?
      tPOS=tPOS+1
      --print('t+='..tPOS)
      if (tPOS) > batchNoPOS then
         tPOS=0
         tNEG=tNEG-1  --cancel index update of negative data because of loop break.
         shufflePOSflag=true
         epochPOS=epochPOS+1
         break
      end
      -- create batch for positive data
      local idx = 1
      for i = 1+batchPOS*(tPOS-1),batchPOS*(tPOS) do
         local img = trainDataPOS.data[shufflePOS[i]]:resize(1,1,12,12) --torch squeezes automatically???? 
	 local imgfl=img:index(dimension,k)
	 xPOS[idx]= torch.cat(img,imgfl,dimension)
         ytPOS[idx] = trainDataPOS.labels[shufflePOS[i]]
         idx = idx + 1
      end
      -- create batch for negative data
      local idx = 1
      for i = 1+batchNEG*(tNEG-1),batchNEG*(tNEG) do
         local img = trainDataNEG.data[shuffleNEG[i]]:resize(1,1,12,12)
	 local imgfl=img:index(dimension,k);
	 xNEG[idx]= torch.cat(img,imgfl,dimension);
         ytNEG[idx] = trainDataNEG.labels[shuffleNEG[i]]
         idx = idx + 1
      end
      --concat the data here
      local shuffle=torch.randperm(opt.batchSize)      
      local xtemp=torch.cat(xPOS:float(),xNEG:float(),1)
      local yttemp=torch.cat(ytPOS:float(),ytNEG:float(),1)
      -- create mini batch
     --print(xtemp:size())
--print(x:size())
      for i = 1,opt.batchSize do
         x[i] = xtemp[shuffle[i]]
         yt[i] = yttemp[shuffle[i]]
      end
         if opt.type == 'cuda' then 
         x = x:cuda()
         yt = yt:cuda()
         end
      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(x)
         local E = loss:forward(y,yt)

         -- estimate df/dW
         local dE_dy = loss:backward(y,yt)   
         model:backward(x,dE_dy)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(y[i],yt[i])
         end

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optimMethod(eval_E, w, optimState)
   end---------------------------------------------------------------------------END TODO batch loop-------------------------------------

   -- time taken
   time = sys.clock() - time
   --print("\n==> time = " .. (time) .. 's')
   print("==> time = " ..time.. 's')
   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   model1 = model:clone()
   --netLighter(model1) --
   torch.save(filename, model1)

   -- next epoch
   confusion:zero()

end--==================================================END_TRAIN===========================

-- Export:
return train
