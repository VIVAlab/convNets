--require('mobdebug').start()
require 'image'   -- to visualize the dataset
require 'math'
----------------------------------------------------------------------

function loadDataFiles(class_dir)
    local i,t,popen = 0, {}, io.popen  
    for filename in popen('ls -A "'..class_dir..'"' ):lines() do
	i = i + 1
	t[i]=class_dir..filename
    end 
    return t, i
end

function ShuffleAndDivideSets(List,SizeImageList)
  local shuffle=torch.randperm(SizeImageList)
  local MaxSize=math.min(SizeImageList,opt.TotNumberperLbl)
  local TrainSize=math.ceil((1-opt.setSplit)*MaxSize)
  local TestSize=MaxSize-TrainSize
  local trainList={}
  local testList={}
  for i=1,TrainSize do
  trainList[i]=List[shuffle[i]]
  end
  for i=TrainSize+1,MaxSize do
  testList[i-TrainSize]=List[shuffle[i]]
  end
return trainList, testList, TrainSize, TestSize
end
-----------------------------------------------------------------------


local desImaX = 12  --Image Width
local desImaY = 12  --Image Height
local ivch = 1
local numblbls=45 -- eventually... put a function that counts the number of folders to make the numblbls reading automatic
local crdnlty=torch.Tensor(numblbls,2)-- to store training data lengths and testdata lengths for each label respectively [#trainData_i,#testData_i],i in [1,45]
local config = require('../config')
path = config.calibDataCropped
---------loop to load ALL data
for lbl=1,numblbls do --labels
imageslist, SizeImageList = loadDataFiles(path..lbl..'/')
imageslist, imageslistt, crdnlty[{lbl,1}], crdnlty[{lbl,2}] = ShuffleAndDivideSets(imageslist,SizeImageList)

	if lbl==1 then
	
		      trdata = torch.Tensor(crdnlty[{lbl,1}], ivch, desImaX, desImaY):fill(0)
		      trlabels = torch.Tensor(crdnlty[{lbl,1}]):fill(lbl)
		      trsize = crdnlty[{lbl,1}] 
		   
	 
		      tedata = torch.Tensor(crdnlty[{lbl,2}], ivch, desImaX, desImaY):fill(0)
		      telabels = torch.Tensor(crdnlty[{lbl,2}]):fill(lbl)
		      tesize = crdnlty[{lbl,2}]

		for j,filename in ipairs(imageslist) do
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			trdata[j] = image.rgb2y(im)
		end
		imageslist = nil
   		print('train data loaded for label '..lbl)
	
		for j,filename in ipairs(imageslistt) do
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			tedata[j] = image.rgb2y(im)
		end
		print('test data loaded for label '..lbl)		
	   	imageslistt = nil

	else
		
	      	trdata = torch.cat(trdata,torch.Tensor(crdnlty[{lbl,1}], ivch, desImaX, desImaY):fill(0),1)
	      	trlabels = torch.cat(trlabels,torch.Tensor(crdnlty[{lbl,1}]):fill(lbl),1)
	      	trsize = trlabels:size()[1]
	   

	      	tedata = torch.cat(tedata,torch.Tensor(crdnlty[{lbl,2}], ivch, desImaX, desImaY):fill(0),1)
	      	telabels = torch.cat(telabels,torch.Tensor(crdnlty[{lbl,2}]):fill(lbl),1)
	      	tesize = telabels:size()[1]
			   
		for j,filename in ipairs(imageslist) do
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			trdata[j+trsize-crdnlty[{lbl,1}]] = image.rgb2y(im)
			
		end
		imageslist = nil
   		print('train data loaded for label '..lbl)

	
		for j,filename in ipairs(imageslistt) do
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			tedata[j+tesize-crdnlty[{lbl,2}]] = image.rgb2y(im)
		end
		imageslistt = nil
   		print('test data loaded for label '..lbl)

	--NOTE: memory intensive 45 labels for large images(note large in this specific case) is very memory intensive consider partitioning
	end
	
end


trainData = {
		      data=trdata,
		      labels=trlabels,
		      size = function() return trlabels:size()[1] end
		   }
	 testData = {
		      data=tedata,
		      labels=telabels,
		      size = function() return telabels:size()[1] end
		   }
			trdata = nil
		      	trlabels = nil
		      	tedata = nil
		      	telabels = nil
---------

trSize=trainData.labels:size()[1]
teSize=testData.labels:size()[1]

-- Displaying the dataset architecture ---------------------------------------
print(sys.COLORS.red ..  'Training Data:')
print(trainData)
print()

print(sys.COLORS.red ..  'Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'
print('preprocessing done')

trainData.size = function() return trSize end  
testData.size = function() return teSize end	

-- classes: GLOBAL var!
--classes = {'face','backg'}  --TODO change to {'1','2',3',...,'45'} automatically
classes = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45'}
-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
