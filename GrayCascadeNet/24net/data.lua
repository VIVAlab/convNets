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
			  --break
			  return t,i
		end
	       t[i] = nfd_j..filename
	    end
	end
    local nBckg = i;	--number of backgrounds
    return t,nBckg
end

function loadDataFilesTEST(face_dir,nonface_dir)
    local i,t, popen = 0,{}, io.popen  
	for j,fd_j in ipairs(face_dir) do
	    for filename in popen('ls -A "'..fd_j..'"' ):lines() do
		   i = i + 1
	       t[i] = fd_j..filename
	    end
	end
    local nFaces = i;	--number of faces
	for j,nfd_j in ipairs(nonface_dir) do
	    for filename in popen('ls -A "'..nfd_j..'"' ):lines() do
		i = i + 1	
			if (i-nFaces>opt.backgroundNumber) then
			  i=i-1;	
			  --break
			  return t,nFaces,i
			end
		t[i] = nfd_j..filename
	    end	
	end
    return t,nFaces,i
end

local config = require('../config')
POStrainadresses = config.positiveTrain
testadresslist = config.positiveTest
negtest = config.negativeTest
negtrain = config.negativeTrain

imageslistPOS,FaceNo = loadDataFilesPOS(POStrainadresses) --Positive Train Data Load
imageslistNEG,BckgNo = loadDataFilesNEG(negtrain) --Positive Train Data Load
imageslistt,lt,teSize = loadDataFilesTEST(testadresslist,negtest) -- Test Data Load
local desImaX = 24  --Image Width
local desImaY = 24  --Image Height
local duplicfliplrTrain='false'  -- set to 'true' if we add flipped data, else 'false' to use only the original data 
local duplicfliplrTest='false'
local ivch = 1
local labelFace = 1 -- label for person and background:
local labelBg = 2 


if (duplicfliplrTrain=='false') then
	trainDataPOS = {
	      data = torch.Tensor(FaceNo, ivch, desImaX, desImaY),
	      labels = torch.Tensor(FaceNo),
	      size = function() return FaceNo end
	   }
	trainDataNEG = {
	      data = torch.Tensor(BckgNo, ivch, desImaX, desImaY),
	      labels = torch.Tensor(BckgNo),
	      size = function() return BckgNo end
	   }
	for j,filename in ipairs(imageslistPOS) do
			print(filename)
			local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataPOS.data[j] = im
		trainDataPOS.labels[j] = labelFace;
	end
	for j,filename in ipairs(imageslistNEG) do
			print(filename)
			local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataNEG.data[j] = im
		trainDataNEG.labels[j] = labelBg;
	end

  elseif  (duplicfliplrTrain=='true') then
	trainDataPOS = {
	      data = torch.Tensor(2*FaceNo, ivch, desImaX, desImaY),
	      labels = torch.Tensor(2*FaceNo),
	      size = function() return 2*FaceNo end
	   }
	trainDataNEG = {
	      data = torch.Tensor(BckgNo, ivch, desImaX, desImaY),
	      labels = torch.Tensor(BckgNo),
	      size = function() return BckgNo end
	   }
	for j,filename in ipairs(imageslistPOS) do  --<=== CRASH HERE DATA EXPECTED GOT NIL
		print(filename)
		local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataPOS.data[j] = im
		image.hflip(trainDataPOS.data[j+FaceNo], im)
		trainDataPOS.labels[j] = labelFace;
		trainDataPOS.labels[j+FaceNo] = labelFace;

	 end
	 for j,filename in ipairs(imageslistNEG) do
			print(filename)
			local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
		trainDataNEG.data[j] = im
		trainDataNEG.labels[j] = labelBg;
	 end
  else
    error("invalid operation: duplicfliplrTrain has to be either 'true' or 'false'.")
end
   
  

   imageslistPOS = nil
   imageslistNEG = nil
   print('train data loaded')
   ----------------------------------------------------

if (duplicfliplrTest=='false') then
   print(lt..' '..teSize)	
	   testData = {
	      data = torch.Tensor(teSize, ivch,desImaX,desImaY),
	      labels = torch.Tensor(teSize),
	      size = function() return teSize end
	   }
	for j,filename in ipairs(imageslistt) do
			--print(filename)
			--filename='/home/jblan016/FaceDetection/Cascade/dataset/test/c_faces/pic00001.jpg'
			local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
			testData.data[j] = im
			if(j <= lt) then --if it is a face
				testData.labels[j] = labelFace;
			else  -- if it's a Bg
				testData.labels[j] = labelBg;
			end
	   end
	   print('test data loaded')		
	   imageslistt = nil
elseif (duplicfliplrTest=='true') then
print(2*lt..' '..teSize+lt)
	testData = {
	      data = torch.Tensor(teSize+lt, ivch,desImaX,desImaY),
	      labels = torch.Tensor(teSize+lt),
	      size = function() return teSize+lt end
	   }

	  for j,filename in ipairs(imageslistt) do
		--print(filename)
		local im =  image.load(filename):float()
		if im:size(1)~=1 then
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2y(im)
		else
		im =  image.scale(im,desImaX,desImaY)
		end
		testData.data[j] = im
 		if(j <= lt) then
		image.hflip(testData.data[j+teSize], im)
		testData.labels[j] = labelFace;
		testData.labels[j+teSize] = labelFace;
		else
			testData.labels[j] = labelBg;
		end
	   end
	   print('test data loaded')		
	   imageslistt = nil
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
trainDataPOS.size = function() return FaceNo end
trainDataNEG.size = function() return BckgNo end

elseif (duplicfliplrTrain=='true') then
trainDataPOS.size = function() return 2*FaceNo end
trainDataNEG.size = function() return BckgNo end
end
if (duplicfliplrTest=='false') then
testData.size = function() return teSize end

elseif (duplicfliplrTest=='true') then
testData.size = function() return teSize+lt end
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
