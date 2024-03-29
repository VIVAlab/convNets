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
   -a,--trainingratio      (default 1)           ratio of negative over positive examples in training batch >=1
   -b,--batchSize          (default 128)         batch size
   -c,--setSplit           (default .1)          ratio of #test data/#test+training data
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -f,--fold               (default 1)           fold number
   -g,--optimization       (default 'sgd')       optimization method: choose 'sgd','cg' or 'lbfgs'
   -i,--devid              (default 1)           device ID (if using CUDA)
   -k,--maxIter            (default 1e2)         maximum number of iterations
   -l,--load		       (default "")          load old model by providing address
      --patches            (default all)         percentage of samples to use for testing'
      --visualize                                visualize dataset
   -m,--momentum           (default 0.1)         momentum
   -n,--CeilNumber         (default 100e3)       background number for training
   -o,--save               (default results)     save directory
   -p,--type               (default float)       float or cuda
   -r,--learningRate       (default 1e-3)        learning rate
   -s,--size               (default small)       dataset: small or full or extra
   -t,--threads            (default 8)           number of threads
   -u,--dropout            (default 0.5)         dropout amount
   -v,--patchsidepercent   (default 1)           length of positive data patch for generalisation. in ]0,1] 0 is pixel sized patch, recommended not below .875     
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -x,--epochstop          (default 400)         stops training when reaches this number of epochs
   -y,--logid              (default 1)           label for the log files
]]

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
local epchcnt=0
while epchcnt<3*opt.epochstop do
if epchcnt <=opt.epochstop then
opt.learningRate=1e-2
elseif epchcnt >opt.epochstop and epchcnt <=2*opt.epochstop then
opt.learningRate=1e-3
elseif epchcnt >2*opt.epochstop and epchcnt <3*opt.epochstop then
opt.learningRate=1e-4
opt.weightDecay=1e-6
end
epchcnt=epchcnt+1

   train(data.trainData)
   test(data.testData)
end
