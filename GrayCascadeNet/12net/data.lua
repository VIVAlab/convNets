require 'image'   -- to visualize the dataset
require 'math'
----------------------------------------------------------------------
  torch.manualSeed(1234)


--io.stdin:read'*l'
function loadDataFiles(dir_list)
    local i,t, popen = 0,{}, io.popen  
	for j,d_j in ipairs(dir_list) do
	    for filename in popen('ls -A "'..d_j..'"' ):lines() do
		   i = i + 1
	       t[i] = d_j..filename
	    end
	end
    local nObjects = i;	--number of images
    return t,nObjects
end

function ShuffleAndDivideSets(List,SizeImageList)
  local MaxSize=math.min(SizeImageList,opt.CeilNumber)
  local shuffle=torch.randperm(MaxSize)
  local TrainSize=math.ceil((1-opt.setSplit)*MaxSize)
  local TestSize=MaxSize-TrainSize
  local masktest = torch.ByteTensor(MaxSize):fill(0)
  masktest:narrow(1,1+(opt.fold-1)*TestSize,TestSize):fill(1)
  local teshuffle=shuffle[masktest]

  local trshuffle=shuffle[torch.add(-masktest,1)]
  local trainList={}
  local testList={}
  for i=1,TrainSize do
    trainList[i]=List[trshuffle[i]]
  end
  for i=1,TestSize do
    testList[i]=List[teshuffle[i]]
  end
return trainList, testList, TrainSize, TestSize
end

function loadscaleimage(imgaddress,H,W)
    local img =  image.load(imgaddress):float()--will return a --I(3xHxW)
    if img:size(1)==1 then
    img = img:repeatTensor(3,1,1)
    elseif img:size(1)==4 then
    print('imgaddress has 4 channels')
    img = img[{{1,3},{},{}}]:clone()
    end
    local s=img:size() --s={ich,H',W'}
      img=image.scale(img,W,H,'bicubic') 
return img
end


-----------------------------------------------------------------------

local Width = 12  --Image Width
local Height =12 --Image Height
-- note no scaling done 
local mdlWidth = 12  --Model input Width
local mdlHeight = 12 --Model input Height
local numblbls = 2
local ich = 1
local datasetdir='/home/jblan016/FaceDetection/Cascade/dataset/'
POSadresses={datasetdir..'AFLW_TrainingTest/',datasetdir..'faces/'}--,datasetdir..'c_faces_train/',datasetdir..'c_faces_test/'}--problem with loading ppm
NEGadresses={datasetdir..'negative/'}
local labelstring={'Faces','Backgrounds'}
---------loop to load ALL data

trdata={}
trlabels={}
tedata={}
telabels={}
local lbl = 1
    imageslist, SizeImageList = loadDataFiles(POSadresses)
    imageslist, imageslistt, trsize, tesize = ShuffleAndDivideSets(imageslist,SizeImageList)

    trdata[lbl] = torch.Tensor(trsize, ich, Height, Width)
    trlabels[lbl] = torch.Tensor(trsize):fill(lbl)

    tedata[lbl] = torch.Tensor(tesize, ich, Height, Width)
    telabels[lbl] = torch.Tensor(tesize):fill(lbl)
	   
    for j,filename in ipairs(imageslist) do
	--print(filename)
      if ich == 1 then
	    trdata[lbl][j] = image.rgb2y(loadscaleimage(filename,Height,Width))
      elseif ich ==3 then
        trdata[lbl][j] = loadscaleimage(filename,Height,Width)
      end
    end
    imageslist = nil
    print('train data loaded for '..labelstring[lbl])


    for j,filename in ipairs(imageslistt) do
        --print(filename)
        if ich == 1 then
	      tedata[lbl][j] = image.rgb2y(loadscaleimage(filename,Height,Width))
        elseif ich ==3 then
          tedata[lbl][j] = loadscaleimage(filename,Height,Width)
        end
    end
    imageslistt = nil
    print('test data loaded for label '..labelstring[lbl])
lbl = 2
    imageslist, SizeImageList = loadDataFiles(NEGadresses)
    imageslist, imageslistt, trsize, tesize = ShuffleAndDivideSets(imageslist,SizeImageList)

    trdata[lbl] = {}
    trlabels[lbl] = torch.Tensor(trsize):fill(lbl)

    tedata[lbl] = {}
    telabels[lbl] = torch.Tensor(tesize):fill(lbl)
	   
    for j,filename in ipairs(imageslist) do
	--print(filename)
	trdata[lbl][j] = image.load(filename):float()
    
      if ich == 1 then
	    trdata[lbl][j] = image.rgb2y(image.load(filename):float())
      elseif ich ==3 then
        trdata[lbl][j] = image.load(filename):float()
      end
    end
    imageslist = nil
    print('train data loaded for '..labelstring[lbl])


    for j,filename in ipairs(imageslistt) do
        --print(filename)
        if ich == 1 then
	    tedata[lbl][j] = image.rgb2y(image.load(filename):float())
      elseif ich ==3 then
        tedata[lbl][j] = image.load(filename):float()
      end

    end
    imageslistt = nil
    print('test data loaded for '..labelstring[lbl])


------------------------------------

trainData = {
	      data=trdata,
	      labels=trlabels
	   }
 testData = {
	      data=tedata,
	      labels=telabels
	   }

trdata = nil
trlabels = nil
tedata = nil
telabels = nil


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

classes = {}
for i=1,numblbls do
classes[i]=''..i..''
end
-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes,
   
}
