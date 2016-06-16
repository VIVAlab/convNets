--require('mobdebug').start()
require 'image'   -- to visualize the dataset
require 'math'
----------------------------------------------------------------------
  torch.manualSeed(1234)


--io.stdin:read'*l'
function loadDataFiles(class_dir)
    local i,t,popen = 0, {}, io.popen  
    for filename in popen('ls -A "'..class_dir..'"' ):lines() do
	i = i + 1
	t[i]=class_dir..filename
    end 
    return t, i
end

function ShuffleAndDivideSets(List,SizeImageList)
  local MaxSize=math.min(SizeImageList,opt.TotNumberperLbl)
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

function loadscaleimage(imgaddress,ich,H,W)
    local img =  image.loadPNG(imgaddress):float()--will return a --I(3xHxW)

    local s=img:size() --s={ich,H',W'}
    if opt.warp=='false' then

      if s[3]/W<s[2]/H then
        img=image.scale(img,W,torch.round(s[2]*W/s[3]),'bicubic')
      elseif s[3]/W>=s[2]/H then
        img=image.scale(img,torch.round(s[3]*H/s[2]),H,'bicubic')
      end
    else 
      img=image.scale(img,W,H,'bicubic') 
    end
  if ich == 1 then
    if img:size(1)==4 then
      --print(imgaddress..'4 channels')
      img = img[{{1,3},{},{}}]:clone()
      img=image.rgb2y(img)
    elseif img:size(1) == 3 then 
      img=image.rgb2y(img)
    end
  end

return img
end



-----------------------------------------------------------------------

local Width = 48  --Image Width
local Height =48 --Image Height
-- note no scaling done 
local mdlWidth = 48  --Model input Width
local mdlHeight = 48 --Model input Height

local ich = 1
local numblbls=45 
classes={}
for i=1,numblbls do
classes[i]=''..i..''
end
---------loop to load ALL data

trdata={}
trlabels={}
tedata={}
telabels={}
trsize={}
tesize={}
imdir='/home/jblan016/FaceDetection/Cascade/dataset/data/cropped/'
for lbl=1,numblbls do --labels
    imageslist, SizeImageList = loadDataFiles(imdir..classes[lbl]..'/')
    imageslist, imageslistt, trsz, tesz = ShuffleAndDivideSets(imageslist,SizeImageList)

    trdata[lbl] = torch.Tensor(trsz, ich, Height, Width)
    trlabels[lbl] = torch.Tensor(trsz):fill(lbl)
    trsize[lbl] = trsz

    tedata[lbl] = torch.Tensor(tesz, ich, Height, Width)
    telabels[lbl] = torch.Tensor(tesz):fill(lbl)
    tesize[lbl] = tesz
	   
    for j,filename in ipairs(imageslist) do
	--print(filename)
	trdata[lbl][j] = loadscaleimage(filename,ich,Height,Width)
    end
    imageslist = nil
    print('train data loaded for label '..lbl)
    for j,filename in ipairs(imageslistt) do
        --print(filename)
        tedata[lbl][j] = loadscaleimage(filename,ich,Height,Width)
    end
imageslistt = nil
print('test data loaded for label '..lbl)
	
end

--PCA--------------------------------

P={}
local X = torch.Tensor()
if ich == 3 then
  local S = torch.Tensor(ich,ich)
  local P1=torch.Tensor()
  local P2=torch.Tensor()
  local P3=torch.Tensor()

  for lbl=1,numblbls do
   P[lbl]={}
   P1 = torch.Tensor(ich,Height,Width):fill(0)
   P2 = torch.Tensor(ich,Height,Width):fill(0)
   P3 = torch.Tensor(ich,Height,Width):fill(0)
   X = torch.Tensor(trsize[lbl],ich)
      for ix=1,mdlWidth do
        for iy=1,mdlHeight do
          for k=1,ich do
            X[{ {},k}] = (trdata[lbl])[{{},k,iy,ix }]-(trdata[lbl])[{{},k,iy,ix }]:mean()
          end
         S = (X:t())*X 
         S= torch.div(S,trsize[lbl]-1) --sample covariance matrix
         U,sigs,V = torch.svd(S)
         P1[{{},iy,ix}]=torch.mul(U[{{},1}],sigs[1])
         P2[{{},iy,ix}]=torch.mul(U[{{},2}],sigs[2])
         P3[{{},iy,ix}]=torch.mul(U[{{},3}],sigs[3])
        end
      end
    P[lbl][1]=P1
    P[lbl][2]=P2
    P[lbl][3]=P3
    P1=nil
    P2=nil
    P3=nil
  print('PCA for label '..lbl..' is done.')
  end
elseif ich == 1 then
  local S = torch.Tensor(ich,ich)
  local P1=torch.Tensor()

  for lbl=1,numblbls do
   P[lbl]={}
   P1 = torch.Tensor(ich,Height,Width):fill(0)

   X = torch.Tensor(trsize[lbl],ich)

      for ix=1,mdlWidth do
        for iy=1,mdlHeight do
          for k=1,ich do
            X[{ {},k}] = (trdata[lbl])[{{},k,iy,ix }]-(trdata[lbl])[{{},k,iy,ix }]:mean()
          end
         S = (X:t())*X 
         S= torch.div(S,trsize[lbl]-1) --sample covariance matrix
         U,sigs,V = torch.svd(S)
         P1[{{},iy,ix}]=torch.mul(U[{{},1}],sigs[1])
        end
      end
    P[lbl][1]=P1
    P1=nil
   print('PCA for label '..lbl..' is done.')
  end
end

print('\nThe PC class conditionals are\n')
print(P)
print()

torch.save('./results/P.dat',P)
U=nil
S=nil
V=nil
X=nil
------------------------------------
---------
trSize=0
teSize=0
for lbl=1,numblbls do
print(trsize[lbl])
trSize=trSize+trsize[lbl]
teSize=teSize+tesize[lbl]
end

trainData = {
	      data=trdata,
	      labels=trlabels,
	      size = function() return trSize end
	   }
 testData = {
	      data=tedata,
	      labels=telabels,
	      size = function() return teSize end
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
trdatamiddle = nil
trainData.size = function() return trSize end  
testData.size = function() return teSize end	


-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes,
   P
   
}
