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
require 'image'
require 'math'
----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
--local fwmodel = t.model
local loss = t.loss
local mdlHeight=12
local mdlWidth=12

--------------------- Backgound scale variable ----------------
MinFaceSize=40
local ich = 3
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
local trainLogger = optim.Logger(paths.concat(opt.save, 'trainAvgValid'..opt.logid..'.log'))

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
local batchNoPOS=torch.floor(((trainData.data[1]:size(1)))/batchPOS)
local batchNoNEG=torch.floor((#(trainData.data[2]))/batchNEG)
print('b#-='..batchNoNEG)
print('b#+='..batchNoPOS)
local xNEG = torch.Tensor(batchNEG,ich,mdlHeight,mdlWidth) 
local xPOS = torch.Tensor(batchPOS,ich,mdlHeight,mdlWidth) 
local x = torch.Tensor(opt.batchSize,ich,mdlHeight,mdlWidth) 

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

local shufflePOS = torch.Tensor((trainData.data[1]):size(1)):fill(0)
local shuffleNEG = torch.Tensor(#(trainData.data[2])):fill(0)
if opt.type == 'cuda' then 
   xNEG=xNEG:cuda()
   xPOS=xPOS:cuda()
   x = x:cuda()

   ytPOS=ytPOS:cuda()
   ytNEG=ytNEG:cuda()
   yt = yt:cuda()
end
----------------------------------------------------------------------
--Data augmentation functions
function randcropscale(img,ich,mdlH,mdlW)
  local  h = img:size(2)
  local  w = img:size(3)
  local  F = torch.floor(torch.rand(1)[1]*math.min(w-MinFaceSize+1,h-MinFaceSize+1))+MinFaceSize
  local offsetH = torch.ceil(torch.rand(1)[1]*(h-F+1))
  local offsetW = torch.ceil(torch.rand(1)[1]*(w-F+1))
  ima=img:sub(1,ich,offsetH,offsetH+F-1,offsetW,offsetW+F-1)
  ima=image.scale(ima,mdlH,mdlW,'bicubic')
  return ima
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
--[[
local L=64
local Lmin=torch.ceil(opt.patchsidepercent*L)
local Lpatch=Lmin
local di=0
local dj=0
]]

local function train(trainData)   ----------------------=================================TRAIN============

   -- epoch tracker
   epochPOS = epochPOS or 1
   epochNEG = epochNEG or 1
   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
	if shufflePOS_FLAG==true then
    shufflePOS = torch.randperm((trainData.data[1]):size(1))
          shufflePOS_FLAG=false
	end
	if shuffleNEG_FLAG==true then
    shuffleNEG = torch.randperm(#(trainData.data[2]))
          shuffleNEG_FLAG=false
	end
   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on positive training data:') 
   print("==> online positive data epoch # " .. epochPOS .. ' [batchSize(+) = ' .. batchPOS .. '].'..' #data(+)='..trainData.data[1]:size(1)) 
   print("==> online negative data epoch # " .. epochNEG .. ' [batchSize(-) = ' .. batchNEG .. '].'..' #data(-)='..#(trainData.data)[2])

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
         xPOS[idx] = trainData.data[1][shufflePOS[i]]--image.scale(trainData.data[1][shufflePOS[i]],mdlHeight,mdlWidth)
         ytPOS[idx] = 1
          if torch.rand(1)[1]>.5 then
          xPOS[idx] = image.hflip(xPOS[idx])
         end
         idx = idx + 1
      end

      -- create batch for negative data
      local idx = 1
      for i = 1+batchNEG*(tNEG-1),batchNEG*(tNEG) do
         xNEG[idx] = randcropscale(trainData.data[2][shuffleNEG[i]],ich,mdlHeight,mdlWidth) ------------------randcropscale(img,ich,mdlH,mdlW)
         ytNEG[idx] = 2
         if torch.rand(1)[1]>.5 then
          xNEG[idx] = image.hflip(xNEG[idx])
         end
         idx = idx + 1
      end
      --concat the data here
      local shuffle=torch.randperm(opt.batchSize)      
      local xtemp=torch.cat(xPOS:float(),xNEG:float(),1)
      local yttemp=torch.cat(ytPOS:float(),ytNEG:float(),1)
	
      -- create mini batch
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
   end--------------------------------------END batch loop-------------------------------------

   -- time taken
   time = sys.clock() - time
   --print("\n==> time = " .. (time) .. 's')
   print("==> time = " ..time.. 's')
   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   if (confusion.averageValid==confusion.averageValid) then
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.averageValid * 100}
   end
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   --commented because we're saving whenever the validation set error was minimal
   --[[-- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   model1 = model:clone()
   --netLighter(model1) --
   torch.save(filename, model1)
--]]
   -- next epoch
   confusion:zero()

end--==================================================END_TRAIN===========================

-- Export:
return train
