require 'image'   -- to visualize the dataset
----------------------------------------------------------------------

function loadDataFilesPOS(face_dir)
    local i,t, popen = 0,{}, io.popen  
	for j,fd_j in ipairs(face_dir) do
	    for filename in popen('ls -A "'..fd_j..'"' ):lines() do
		   i = i + 1
	       t[i] = fd_j..filename
	    end
	end
    local nFaces = i;	--number of faces
    return t,nFaces
end
function loadDataFilesNEG(nonface_dir)
    local i,t, popen = 0,{}, io.popen
	for j,nfd_j in ipairs(nonface_dir) do
	    for filename in popen('ls -A "'..nfd_j..'"' ):lines() do
		   i = i + 1
		if (i>opt.backgroundNumber) then
			  i=i-1;	
			  return t,i
		end
	       t[i] = nfd_j..filename
	    end
	end
    local nBckg = i;	--number of backgrounds
    return t,nBckg
end

function joinPOSNEGseparateTrainTest(imlistPOS,POSno,imlistNEG,NEGno)
  local nPOStest=torch.ceil(POSno*opt.setSplit)
  local nPOStrain=POSno-nPOStest
  
  local nNEGtest=torch.ceil(NEGno*opt.setSplit)
  local nNEGtrain=NEGno-nNEGtest

  --shuffle to mix merged sets
  local shufflePOS=torch.randperm(POSno)
  local shuffleNEG=torch.randperm(NEGno)
  local POStrainlist={}
  local NEGtrainlist={}
  local testList={}
  
  for i=1,nPOStest do
    testList[i]=imlistPOS[shufflePOS[i]]
  end
  for i=nPOStest+1,POSno do
    POStrainlist[i-nPOStest]=imlistPOS[shufflePOS[i]]
  end

  for i=1,nNEGtest do
    testList[nPOStest+i]=imlistNEG[shuffleNEG[i]]
  end
  for i=nNEGtest+1,NEGno do
    NEGtrainlist[i-nNEGtest]=imlistNEG[shuffleNEG[i]]
  end



  return POStrainlist, nPOStrain, NEGtrainlist, nNEGtrain, testList ,nNEGtest , nPOStest  
end

POSadresses={'/home/jblan016/FaceDetection/Cascade/dataset/train/aflw/','/home/jblan016/FaceDetection/Cascade/dataset/train/faces/','/home/jblan016/FaceDetection/Cascade/dataset/train/c_faces/','/home/jblan016/FaceDetection/Cascade/dataset/test/faces/'}--,'/home/jblan016/FaceDetection/Cascade/dataset/train/faces/','/home/jblan016/FaceDetection/Cascade/dataset/train/c_faces/'}--'/home/jblan016/FaceDetection/Cascade/dataset/train/faces/','/home/jblan016/FaceDetection/Cascade/dataset/train/c_faces/'
--testadresslist={'/home/jblan016/FaceDetection/Cascade/dataset/test/c_faces/'}--,'/home/jblan016/FaceDetection/Cascade/dataset/test/faces/'
--negtest={'/home/jblan016/FaceDetection/Cascade/dataset/test/nonfaces/'}--'/home/jblan016/FaceDetection/Cascade/dataset/test/nonfaces/',
NEGadresses={'/home/jblan016/FaceDetection/Cascade/BgGenerator/NegativeData0Full/'}--,'/home/jblan016/FaceDetection/Cascade/BgGenerator/NegativeData0Full/'
--negtrain={'/home/jblan016/FaceDetection/Cascade/BgGenerator/24NetPatches_negatives/','/home/jblan016/FaceDetection/Cascade/BgGenerator/AFLW_FACELESS_PATCHES/'}
imageslistPOS,FaceNo = loadDataFilesPOS(POSadresses) --Positive Train Data Load
imageslistNEG,BckgNo = loadDataFilesNEG(NEGadresses) --Positive Train Data Load
POStrainlist, FaceNotrain, NEGtrainlist ,BckgNotrain ,testList ,BckgNotest ,FaceNotest = joinPOSNEGseparateTrainTest(imageslistPOS,FaceNo,imageslistNEG,BckgNo)
print('Face#='..FaceNo,'Bckg#='..BckgNo)
--print('Face_train#='..FaceNotrain,'Face_test#='..FaceNotest,'bckg#_train#='..BckgNotrain,'bckg#_test#='..BckgNotest)

   imageslistPOS = nil
   imageslistNEG = nil
   FaceNo=nil
   BckgNo=nil
local desImaX = 12  --Image Width
local desImaY = 12  --Image Height
local duplicfliplrTrain='false'  -- set to 'true' if we add flipped data, else 'false' to use only the original data 
local duplicfliplrTest='false'
local ivch = 1 --input channels
local labelFace = 1 -- label for person and background:
local labelBg = 2 
local aflwX=64
local aflwY=64

if (duplicfliplrTrain=='false') then
	trainDataPOS = {
	      data = torch.Tensor(FaceNotrain, ivch, aflwX, aflwY),
	      labels = torch.Tensor(FaceNotrain),
	      size = function() return FaceNotrain end
	   }
	trainDataNEG = {
	      data = torch.Tensor(BckgNotrain, ivch, desImaX, desImaY),
	      labels = torch.Tensor(BckgNotrain),
	      size = function() return BckgNotrain end
	   }
	for j,filename in ipairs(POStrainlist) do
			print(filename)
			local im =  image.load(filename):float()

			if im:size(1)==3 then
		--im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		elseif im:size(1)==1 then
		--im =  image.scale(im,desImaX,desImaY)
		end
		trainDataPOS.data[j] = im
		trainDataPOS.labels[j] = labelFace;
	end
	for j,filename in ipairs(NEGtrainlist) do
			print(filename)
			local im =  image.load(filename):float()
			if im:size(1)==3 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		elseif im:size(1)==1 then
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataNEG.data[j] = im
		trainDataNEG.labels[j] = labelBg;
	end

  elseif  (duplicfliplrTrain=='true') then
	trainDataPOS = {
	      data = torch.Tensor(2*FaceNotrain, ivch, aflwX, aflwY),
	      labels = torch.Tensor(2*FaceNotrain),
	      size = function() return 2*FaceNotrain end
	   }
	trainDataNEG = {
	      data = torch.Tensor(BckgNotrain, ivch, desImaX, desImaY),
	      labels = torch.Tensor(BckgNotrain),
	      size = function() return BckgNotrain end
	   }
	for j,filename in ipairs(POStrainlist) do
		print(filename)
		local im =  image.load(filename):float()
		if im:size(1)==3 and im:size(2)~=24 and im:size(3)~=24 then
		--im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		elseif im:size(1)==1 and im:size(2)==24 and im:size(3)==24 then
		--im =  image.scale(im,desImaX,desImaY)
		end
		trainDataPOS.data[j] = im
		image.hflip(trainDataPOS.data[j+FaceNotrain], im)
		trainDataPOS.labels[j] = labelFace;
		trainDataPOS.labels[j+FaceNotrain] = labelFace;

	 end
	 for j,filename in ipairs(NEGtrainlist) do
			print(filename)
			local im =  image.load(filename):float()
			if im:size(1)==3 and im:size(2)~=24 and im:size(3)~=24 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		elseif im:size(1)==1 and im:size(2)==24 and im:size(3)==24 then
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataNEG.data[j] = im
		trainDataNEG.labels[j] = labelBg;
	 end
  else
    error("invalid operation: duplicfliplrTrain has to be either 'true' or 'false'.")
end
   


   POStrainlist = nil
   NEGtrainlist = nil

   print('train data loaded')
   ----------------------------------------------------

if (duplicfliplrTest=='false') then
   print(FaceNotest..' '..FaceNotest+BckgNotest)	
	   testData = {
	      data = torch.Tensor(FaceNotest+BckgNotest, ivch,desImaX,desImaY),
	      labels = torch.Tensor(FaceNotest+BckgNotest),
	      size = function() return FaceNotest+BckgNotest end
	   }
	for j,filename in ipairs(testList) do
			--print(filename)
			--filename='/home/jblan016/FaceDetection/Cascade/dataset/test/c_faces/pic00001.jpg'
			local im =  image.load(filename):float()
			if im:size(1)==3 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		elseif im:size(1)==1 then
		im =  image.scale(im,desImaX,desImaY)
		end
			testData.data[j] = im
			if(j <= FaceNotest) then --if it is a face
				testData.labels[j] = labelFace;
			else  -- if it's a Bg
				testData.labels[j] = labelBg;
			end
	   end
	   print('test data loaded')		
	   testList = nil
elseif (duplicfliplrTest=='true') then
print(2*FaceNotest..' '..BckgNotest+FaceNotest)
	testData = {
	      data = torch.Tensor(BckgNotest+2*FaceNotest, ivch,desImaX,desImaY),
	      labels = torch.Tensor(BckgNotest+2*FaceNotest),
	      size = function() return BckgNotest+2*FaceNotest end
	   }

	  for j,filename in ipairs(testList) do
		--print(filename)
		local im =  image.load(filename):float()
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		testData.data[j] = im
 		if(j <= FaceNotest) then
		image.hflip(testData.data[j+BckgNotest+FaceNotest], im)
		testData.labels[j] = labelFace;
		testData.labels[j+BckgNotest+FaceNotest] = labelFace;
		else
			testData.labels[j] = labelBg;
		end
	   end
	   print('test data loaded')		
	   testList = nil
else
error("invalid operation: duplicfliplrTest has to be either 'true' or 'false'.")
end
   
-- Displaying the dataset architecture ---------------------------------------
print(sys.COLORS.red ..  'Positive Training Data:')
print(trainDataPOS)
print()
print(sys.COLORS.red ..  'Negative Training Data:')
print(trainDataNEG)
print()
print(sys.COLORS.red ..  'Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'
print('preprocessing done')
if (duplicfliplrTrain=='false') then
trainDataPOS.size = function() return FaceNotrain end
trainDataNEG.size = function() return BckgNotrain end

elseif (duplicfliplrTrain=='true') then
trainDataPOS.size = function() return 2*FaceNotrain end
trainDataNEG.size = function() return BckgNotrain end
end
if (duplicfliplrTest=='false') then
testData.size = function() return BckgNotest+FaceNotest end

elseif (duplicfliplrTest=='true') then
testData.size = function() return BckgNotest+2*FaceNotest end
end
-- classes: GLOBAL var!
classes = {'face','backg'}

-- Exports -------------------------------------------------------------------
return {
   trainDataNEG = trainDataNEG,--TODO add traindataPOS traindataNEG
   trainDataPOS = trainDataPOS,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
