----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'
----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining some tools')

-- model:
local t = require 'model'
local model = t.model
local loss = t.loss
local mdlHeight=48
local mdlWidth=48
--------------------- Backgound scale variable ----------------
MinFaceSize=40
local ich = 3
------------------------------------------------------------
-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)
-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'testAvgValid'..opt.logid..'.log'))
local maxaverageValid=0;--initialisation in percent
-- Batch test:
local inputs = torch.Tensor(opt.batchSize,ich,mdlHeight,mdlWidth)
local targets = torch.Tensor(opt.batchSize)
--------------------------------------------------------
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
function randBSC(imag)  --random brightness/saturation/constrast of image
  local c = (torch.round(torch.mul(torch.rand(3),2))+1)/2
  local g = image.rgb2y(imag)
  local Gm = torch.Tensor(imag:size()):fill(g:mean())
  g = torch.repeatTensor(g,3,1,1)
  imag = torch.mul(imag,c:mean())+torch.mul(Gm,(1-c[1]))+torch.mul(g,(1-c[2]))
  return imag
end
--------------------
--debug function
function pause()
print('press enter to continue')
io.stdin:read'*l'
end
---------------------------
-- cat labels
datalabels=torch.cat(testData.labels[1],testData.labels[2])
testDataSize=(#datalabels)[1]
notestFaces=(#testData.labels[1])[1]
print(testDataSize)
----------------------------------------------------------------------
print(sys.COLORS.red .. '==> defining test procedure')
local batchSize = opt.batchSize
-- Batch test:
local inputs = torch.Tensor(batchSize,ich,mdlHeight, mdlWidth) -- get size from data
local targets = torch.Tensor(batchSize)
-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

   -- test over test data
   print(sys.COLORS.red .. '==> testing on test set:')
   for t = 1,testDataSize,opt.batchSize do
      -- disp progress
      xlua.progress(t, testDataSize)

      -- batch fits?
      if (t + batchSize - 1) > testDataSize then
         --batchSize = testDataSize+1-t
         --inputs = torch.Tensor(batchSize,ich,mdlHeight, mdlWidth) -- get size from data
         --targets = torch.Tensor(batchSize)
         break
      end

      -- create mini batch
      local idx = 1
      for i = t,t+batchSize-1 do
         targets[idx] = datalabels[i]
         if targets[idx]==1 then
         inputs[idx] = testData.data[1][i]
         else

         inputs[idx] = testData.data[2][i-notestFaces]--randcropscale(testData.data[2][i-notestFaces],ich,mdlHeight,mdlWidth)
         end
         idx = idx + 1
      end
      if opt.type == 'cuda' then 
        inputs = inputs:cuda()
        targets = targets:cuda()
      end
      -- test sample
      model:evaluate()
      local preds = model:forward(inputs)
      model:training()
--print(targets[1])	
--print(math.exp(preds[1][1]))
--print(math.exp(preds[1][2]))
      -- confusion
      for i = 1,batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testDataSize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion) --sys.COLORS.red ..
   
   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.averageValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   if maxaverageValid<confusion.averageValid * 100 then
     maxaverageValid=confusion.averageValid * 100
     local filename = paths.concat(opt.save, 'model.net')
     os.execute('mkdir -p ' .. sys.dirname(filename))
     print(sys.COLORS.blue ..'==> saving model to '..filename)
     model1 = model:clone()
     --netLighter(model1) --
     torch.save(filename, model1)
   end
print(' + best average Valid: '..maxaverageValid..'%')

   confusion:zero()
   
end

-- Export:
return test
