------------------------------------------------------------------------------
-- Preprocessing to apply to each dataset
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013, E. Culurciello June 2014
------------------------------------------------------------------------------

print(sys.COLORS.red ..  '==> preprocessing data')

local channels = {'r','g','b'}

local Width = 12  
local Height = 12  
-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')

local mean = {}
local std = {}
local numlbls=45
local cnt=0

-- compute means/stds
for i,channel in ipairs(channels) do
  mean[i]=0
  std[i]=0
  for j = 1, numlbls do
     mean[i] = mean[i]+trainData.data[j][{{},i,{},{} }]:mean()
     std[i]  = mean[i]+trainData.data[j][{{},i,{},{} }]:std()
  end
  mean[i]=mean[i]/numlbls
  std[i]=std[i]/numlbls
end

-- Normalize each channel globally:
for i,channel in ipairs(channels) do
  for j = 1, numlbls do
     trainData.data[j][{ {},i,{},{} }]:add(-mean[i])
     trainData.data[j][{ {},i,{},{} }]:div(std[i])
  end
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
  for j = 1, numlbls do
     testData.data[j][{ {},i,{},{} }]:add(-mean[i])
     testData.data[j][{ {},i,{},{} }]:div(std[i])
  end
end



----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics:')



for i,channel in ipairs(channels) do
  trainMean=0
  trainStd=0
  for j = 1, numlbls do
     trainMean=trainMean+trainData.data[j][{{},i,{},{} }]:mean()
     trainStd=trainStd+trainData.data[j][{{},i,{},{} }]:mean()
  end
  trainMean=trainMean/numlbls
  trainStd=trainStd/numlbls

  testMean=0
  testStd=0
  for j = 1, numlbls do
     testMean=testMean+testData.data[j][{{},i,{},{} }]:mean()
     testStd=testStd+testData.data[j][{{},i,{},{} }]:mean()
  end
  testMean=testMean/numlbls
  testStd=testStd/numlbls

   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)

  -- print('       prenormalized training data, '..channel..'-channel, mean:               ' .. mean[i])
  -- print('       prenormalized test data, '..channel..'-channel, standard deviation: ' .. std[i])
end


torch.save('results/mean.dat',mean)
torch.save('results/std.dat',std)
----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data:')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   -- Showing some training exaples
   local first128Samples = trainData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some training examples'}
   -- Showing some testing exaples
   local first128Samples = testData.data[{ {1,128} }]
   image.display{image=first128Samples, nrow=16, legend='Some testing examples'}
end
