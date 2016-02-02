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
   -a,--trainingratio      (default 3)           ratio of negative over positive examples in training batch >=1
   -g,--optimization       (default 'sgd')       optimization method: choose 'sgd','cg' or 'lbfgs'
   -k,--maxIter            (default 1e2)         maximum number of iterations
   -r,--learningRate       (default 1e-3)        learning rate
   -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
   -w,--weightDecay        (default 1e-5)        L2 penalty on the weights
   -m,--momentum           (default 0.1)         momentum
   -d,--dropout            (default 0.5)         dropout amount
   -b,--batchSize          (default 128)         batch size
   -t,--threads            (default 8)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
   -s,--size               (default small)       dataset: small or full or extra
   -o,--save               (default results)     save directory
   -l,--load		   (default "")          load old model by providing address
      --patches            (default all)         percentage of samples to use for testing'
      --visualize                                visualize dataset
   -n,--backgroundNumber  (default 100e3)       background number for training
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

while true do
   train(data.trainDataPOS,data.trainDataNEG)
   test(data.testData)
end