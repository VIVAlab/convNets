------------------------------------------------------------------------------
-- Preprocessing to apply to each dataset
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013, E. Culurciello June 2014
------------------------------------------------------------------------------

print(sys.COLORS.red ..  '==> preprocessing data')

local channels = {'y'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')
local mean = {}
local std = {}
local meanNEG = {}
local stdNEG = {}
local meanPOS = {}
local stdPOS = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:

   meanNEG[i] = trainDataNEG.data[{ {},i,{},{} }]:mean()
   meanPOS[i] = trainDataPOS.data[{ {},i,{},{} }]:mean()
   mean[i]=(meanPOS[i]+opt.trainingratio*meanNEG[i])/(1+opt.trainingratio)
   trainDataNEG.data[{ {},i,{},{} }]:add(-mean[i])
   trainDataPOS.data[{ {},i,{},{} }]:add(-mean[i])
   std[i]=math.sqrt((trainDataPOS.data[{ {},i,{},{} }]:clone():pow(2):mean()+opt.trainingratio*trainDataNEG.data[{ {},i,{},{} }]:clone():pow(2):mean())/(1+opt.trainingratio))
trainDataNEG.data[{ {},i,{},{} }]:div(std[i])
trainDataPOS.data[{ {},i,{},{} }]:div(std[i])
end
print(meanNEG,meanPOS,mean,std)
-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:

   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end
torch.save('results/mean().dat',mean)
torch.save('results/std().dat',std)
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics:')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMeanPOS = trainDataPOS.data[{ {},i }]:mean()
   local trainStdPOS = trainDataPOS.data[{ {},i }]:std()
   local trainMeanNEG = trainDataNEG.data[{ {},i }]:mean()
   local trainStdNEG = trainDataNEG.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('       positive training data, '..channel..'-channel, mean:               ' .. trainMeanPOS)
   print('       positive training data, '..channel..'-channel, standard deviation: ' .. trainStdPOS)

   print('       negative training data, '..channel..'-channel, mean:               ' .. trainMeanNEG)
   print('       negative training data, '..channel..'-channel, standard deviation: ' .. trainStdNEG)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data:')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainDataPOS.data[{ {1,32} }]
   image.display{image=first32Samples, nrow=16, legend='Some positive training examples'}
   local first128Samples = trainDataNEG.data[{ {1,96} }]
   image.display{image=first96Samples, nrow=16, legend='Some negative training examples'}
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end
