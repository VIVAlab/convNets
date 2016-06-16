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
print('Train')
----------------------------------------------------------------------
-- Model + Loss:
local t = require 'model'
local model = t.model
local fwmodel = t.model
local loss = t.loss

-- model input dimensions
local mdlH=48
local mdlW=48
local ich=3
--[[
local network1 = torch.load('/home/jblan016/FaceDetection/12netmodGray/results/model(BestSGD).net')
local network0 = nn.Sequential()
	for i=1,3 do
		network0:add(network1.modules[i])
	end

if opt.type == 'cuda' then
   network0:cuda()
end
--]]
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
local trainLogger = optim.Logger(paths.concat(opt.save, 'train_'..opt.fold..'V'..opt.logid..'.log'))
local trainvldLogger = optim.Logger(paths.concat(opt.save, 'trainvld_'..opt.fold..'V'..opt.logid..'.log'))
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
else
error('unknown optimization method')
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')
local x = torch.Tensor(opt.batchSize,ich, 
         mdlH, mdlW) --faces data
local yt = torch.Tensor(opt.batchSize)

--[[
if opt.type == 'cuda' then 
   x = x:cuda()
   yt = yt:cuda()
end
--]]
------------------------------------------------------------
--unwrap labels
local flippedlabels = torch.LongTensor(45):fill(0)
dims=5
dimtx=3
dimty=3
sT=torch.Tensor(dims,dimtx,dimty):stride()
for i=1,dims do
  for j=1,dimtx do
    for k=1,dimty do
       flippedlabels[1+(i-1)*sT[1]+(j-1)*sT[2]+(k-1)*sT[3]]=1+(i-1)*sT[1]+(dimtx-j)*sT[2]+(k-1)*sT[3]
    end
  end
end

local indx = torch.LongTensor((#trainData.labels[1])[1],2)

local indxtmp = torch.LongTensor()
indx[{{},1}] = trainData.labels[1]
indx[{{},2}] = torch.range(1,(#trainData.labels[1])[1]):type('torch.LongTensor')

for i=2,#trainData.labels do
  indxtmp = torch.LongTensor((#trainData.labels[i])[1],2)
  indxtmp[{{},1}] = trainData.labels[i]
  indxtmp[{{},2}] = torch.range(1,(#trainData.labels[i])[1]):type('torch.LongTensor')
  indx=torch.cat(indx,indxtmp,1)
end
indxtmp = nil
--Data augmentation functions
function randBSC(imag)  --random brightness/saturation/constrast of image
  local c = (torch.round(torch.mul(torch.rand(3),2))+1)/2
  local g = image.rgb2y(imag)
  local Gm = torch.Tensor(imag:size()):fill(g:mean())
  g = torch.repeatTensor(g,3,1,1)
  imag = torch.mul(imag,c:mean())+torch.mul(Gm,(1-c[1]))+torch.mul(g,(1-c[2]))
  return imag
end

function randlighting(imag,lbl)
  local r=torch.randn(3)*0.316227; -- r~N(0,sqrt(.1)I)
  imag= imag+torch.mul(P[lbl][1],r[1])+torch.mul(P[lbl][2],r[2])+torch.mul(P[lbl][3],r[3])
return imag
end

function randcropscale(img,ich,mdlH,mdlW)
  local  h = img:size(2)
  local  w = img:size(3)
  local a = math.abs(torch.rand(1)[1]+torch.rand(1)[1]-1)
  if h/mdlH<w/mdlW then
  w=torch.round((1-a)*w+a*w*mdlH/h)
  h=torch.round((1-a)*h+a*mdlH)
  elseif h/mdlH>=w/mdlW then
  w=torch.round((1-a)*w+a*mdlW)
  h=torch.round((1-a)*h+a*h*mdlW/w)
  end
  img=image.scale(img,w,h,'bicubic')
  local offsetH=torch.ceil((torch.abs(mdlH-h)+1)*torch.rand(1)[1])
  local offsetW=torch.ceil((torch.abs(mdlW-w)+1)*torch.rand(1)[1])
  local ima = torch.Tensor(ich,mdlH,mdlW):float()
  ima=img:sub(1,ich,offsetH,offsetH+mdlH-1,offsetW,offsetW+mdlW-1)
  return ima
end



----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')
--local epoch

local function train(trainData)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print(sys.COLORS.green .. '==> doing epoch on training data:') 
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())
      collectgarbage()

      -- batch fits?
      if (t + opt.batchSize - 1) > trainData:size() then
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+opt.batchSize-1 do
         yt[idx] = indx[shuffle[i]][1]
         --x[idx] = randlighting( randBSC( randcropscale(trainData.data[yt[idx]][indx[shuffle[i]][2]],ich,mdlH,mdlW) ) ,indx[shuffle[i]][1])
         x[idx] = randBSC( randlighting(trainData.data[yt[idx]][indx[shuffle[i]][2]],indx[shuffle[i]][1]) ) 
         --x[idx] = trainData.data[yt[idx]][indx[shuffle[i]][2]]
         --flipping
         
         if torch.rand(1)[1]>.5 then
          yt[idx] = flippedlabels[yt[idx]]
          x[idx] = image.hflip(x[idx])
         end
         idx = idx + 1
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
      if opt.type == 'cuda' then 
        x = x:float()
        yt = yt:float()
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.averageValid * 100}
   trainvldLogger:add{['% total class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
--[[
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   model1 = model:clone()
   --netLighter(model1)
   torch.save(filename, model1)
--]]
   -- next epoch
   confusion:zero()
   epoch = epoch + 1
   return epoch
end

-- Export:

return train
