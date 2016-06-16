----------------------------------------------------------------------
-- Train a ConvNet as people detector
--
-- E. Culurciello 
-- Mon June 10 14:58:50 EDT 2014
----------------------------------------------------------------------

require 'pl'
require 'trepl'
require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> processing options')

opt = lapp[[
   -g,--optimization       (default 'sgd')       optimization method: choose 'sgd' or 'cg'
   -e,--warp               (default 'true')     warping of training images to a fixed size
   -k,--maxIter            (default 1e2)         maximum number of iterations
   -c,--setSplit           (default 0.1 )         ratio of #test data/#test+training data
   -f,--fold               (default 1)           current fold number
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -n,--TotNumberperLbl    (default 25e3)        background number for training
   -s,--size               (default small)       dataset: small or full or extra
   -l,--load               (default "")          load old model by providing address
   -v,--saveres            (default 1)           save if 1
   -o,--save               (default results)     save directory
      --patches            (default all)         percentage of samples to use for testing'
      --visualize                                visualize dataset
   -x,--epochstop          (default 300)         stops training when reaches this number of epochs
   -y,--logid              (default 1)           label for the log files
]]
---l,--load		   (default "")          load old model by providing address
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- type:
if opt.type == 'cuda' then
   print(sys.COLORS.red ..  '==> switching to CUDA')
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print(sys.COLORS.red ..  '==> using GPU #' .. cutorch.getDevice())
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> load modules')

local data  = require 'data'
local train = require 'train'
local test  = require 'test'

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> training!')
epoch = 0
while true do
   if epoch<40 then
     opt.learningRate=1e-2
   elseif epoch>=40 and epoch<80 then
     opt.learningRate=1e-3
   elseif epoch>=80 then
     opt.learningRate=1e-4
   end
   epoch = train(data.trainData,P)
   test(data.testData)
   if epoch>opt.epochstop then
   break
   end
end
