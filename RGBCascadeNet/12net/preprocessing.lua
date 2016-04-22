------------------------------------------------------------------------------
-- Preprocessing to apply to each dataset
------------------------------------------------------------------------------
-- Alfredo Canziani May 2013, E. Culurciello June 2014
------------------------------------------------------------------------------

print(sys.COLORS.red ..  '==> preprocessing data')

local channels = {'r','g','b'}

local Width = 12  --Image Width
local Height =12  --Image Height
-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: global normalization:')

local mean = {}
local std = {}
local lbl = 0
local cnt = 0
local mean2 = 0
local std2 = 0

-- compute means/stds
for i,channel in ipairs(channels) do
  mean[i]=0
  std[i]=0
  mean2 = 0
  std2 = 0
  cnt = 0
    lbl = 2
     for k=1, #trainData.data[lbl] do
     mean2=mean2+trainData.data[lbl][k][{ i,{},{} }]:mean()
     std2=std2+trainData.data[lbl][k][{ i,{},{} }]:std()
     cnt=cnt+1
     end

    lbl = 1
     mean[i]=trainData.data[lbl][{{},i,{},{} }]:mean()
     std[i]=trainData.data[lbl][{{}, i,{},{} }]:std()

  mean[i]=(trainData.data[lbl]:size(1)*mean[i]+mean2)/(cnt+trainData.data[lbl]:size(1))
  std[i]=(trainData.data[lbl]:size(1)*std[i]+std2)/(cnt+trainData.data[lbl]:size(1))
end

-- Normalize each channel globally:
for i,channel in ipairs(channels) do
    lbl = 1
     trainData.data[lbl][{{},i,{},{} }]:add(-mean[i])
     trainData.data[lbl][{{},i,{},{} }]:div(std[i])
    lbl = 2
     for k=1, #trainData.data[lbl] do
     trainData.data[lbl][k][{ i,{},{} }]:add(-mean[i])
     trainData.data[lbl][k][{ i,{},{} }]:div(std[i])
     end
  
end

-- Normalize test data, using the training means/stds

for i,channel in ipairs(channels) do
    lbl = 1
     testData.data[lbl][{{},i,{},{} }]:add(-mean[i])
     testData.data[lbl][{{},i,{},{} }]:div(std[i])
    lbl = 2
     for k=1, #testData.data[lbl] do
     testData.data[lbl][k][{ i,{},{} }]:add(-mean[i])
     testData.data[lbl][k][{ i,{},{} }]:div(std[i])
     end
end


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics:')



for i,channel in ipairs(channels) do
  trainMean=0
  trainStd=0
  trainMean2 = 0
  trainStd2 = 0
  cnt = 0
    lbl = 2
     for k=1,#trainData.data[lbl] do
     trainMean2=trainMean2+trainData.data[lbl][k][{ i,{},{} }]:mean()
     trainStd2=trainStd2+trainData.data[lbl][k][{ i,{},{} }]:std()
     cnt=cnt+1
     end

    lbl = 1
     trainMean = trainData.data[lbl][{{},i,{},{} }]:mean()
     trainStd = trainData.data[lbl][{{},i,{},{} }]:std()

  trainMean=(trainData.data[lbl]:size(1)*trainMean+trainMean2)/(cnt+trainData.data[lbl]:size(1))
  trainStd=(trainData.data[lbl]:size(1)*trainStd+trainStd2)/(cnt+trainData.data[lbl]:size(1))

  testMean=0
  testStd=0
  testMean2 = 0
  testStd2 = 0
  cnt = 0
    lbl = 2
     for k=1, #testData.data[lbl] do
     testMean2=trainMean2+testData.data[lbl][k][{ i,{},{} }]:mean()
     testStd2=trainStd2+testData.data[lbl][k][{ i,{},{} }]:std()
     cnt=cnt+1
     end

    lbl = 1
     testMean = testData.data[lbl][{{},i,{},{} }]:mean()
     testStd = testData.data[lbl][{{},i,{},{} }]:std()

  testMean = (testData.data[lbl]:size(1)*testMean+testMean2)/(cnt+testData.data[lbl]:size(1))
  testStd = (testData.data[lbl]:size(1)*testStd+testStd2)/(cnt+testData.data[lbl]:size(1))


   print('       training data, '..channel..'-channel, mean:               ' .. trainMean)
   print('       training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('       test data, '..channel..'-channel, mean:                   ' .. testMean)
   print('       test data, '..channel..'-channel, standard deviation:     ' .. testStd)

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
