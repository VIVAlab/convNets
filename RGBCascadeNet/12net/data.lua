require 'image'   -- to visualize the dataset
----------------------------------------------------------------------

function loadDataFiles(face_dir,nonface_dir)
    local i,t, popen = 0,{}, io.popen  
	for j=1,3 do
		if face_dir[j]==nil then
			break
		else
		
		    for filename in popen('ls -A "'..face_dir[j]..'"' ):lines() do
			   i = i + 1
		       t[i] = face_dir[j]..filename
		    end
		end
	end
    local nFaces = i;	--number of faces
    for filename in popen('ls -A "'..nonface_dir..'"' ):lines() do
	i = i + 1	
		if (i-nFaces>opt.backgroundNumber) then
		  i=i-1;	
		  break
		end
        t[i] = nonface_dir..filename
    end	
    return t,nFaces,i
end
trainadresslist={'/home/jblan016/FaceDetection/dataset/train/aflw/','/home/jblan016/FaceDetection/dataset/train/faces/','/home/jblan016/FaceDetection/dataset/train/c_faces/'}
testadresslist={'/home/jblan016/FaceDetection/dataset/test/c_faces/'}
imageslist,l,trSize = loadDataFiles(trainadresslist,'/home/jblan016/FaceDetection/dataset/train/nonfaces/') --Train Data Load
imageslistt,lt,teSize = loadDataFiles(testadresslist,'/home/jblan016/FaceDetection/dataset/test/nonfaces/') -- Test Data Load
local desImaX = 12  --Image Width
local desImaY = 12  --Image Height
local duplicfliplrTrain='true'  -- set to 'true' if we add flipped data, else 'false' to use only the original data 
local duplicfliplrTest='true'
local ivch = 3
local labelFace = 1 -- label for person and background:
local labelBg = 2 


if (duplicfliplrTrain=='false') then
	trainData = {
	      data = torch.Tensor(trSize, ivch, desImaX, desImaY),
	      labels = torch.Tensor(trSize),
	      size = function() return trSize end
	   }
	for j,filename in ipairs(imageslist) do
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			trainData.data[j] = image.rgb2yuv(im)
	 		if(j <= l) then
				trainData.labels[j] = labelFace;
			else
				trainData.labels[j] = labelBg;
			end
	   end
  elseif  (duplicfliplrTrain=='true') then
	trainData = {
	      data = torch.Tensor(trSize+l, ivch, desImaX, desImaY),
	      labels = torch.Tensor(trSize+l),
	      size = function() return trSize+l end
	   }
	for j,filename in ipairs(imageslist) do
		print(filename)
		local im =  image.load(filename):float()
		im =  image.scale(im,12,12)
		im=image.rgb2yuv(im)
		trainData.data[j] = im
 		if(j <= l) then
		image.hflip(trainData.data[j+trSize], im)
		trainData.labels[j] = labelFace;
		trainData.labels[j+trSize] = labelFace;
		else
			trainData.labels[j] = labelBg;
		end
	   end
  else
    error("invalid operation: duplicfliplrTrain has to be either 'true' or 'false'.")
end
   
  

   imageslist = nil
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
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY) --test data
			testData.data[j] = image.rgb2yuv(im)
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
		print(filename)
		local im =  image.load(filename):float()
		im =  image.scale(im,desImaX,desImaY)
		im=image.rgb2yuv(im)
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
print(sys.COLORS.red ..  'Training Data:')
print(trainData)
print()

print(sys.COLORS.red ..  'Test Data:')
print(testData)
print()

-- Preprocessing -------------------------------------------------------------
dofile 'preprocessing.lua'
print('preprocessing done')
if (duplicfliplr=='false') then
trainData.size = function() return trSize end
testData.size = function() return teSize end
elseif (duplicfliplr=='true') then
trainData.size = function() return trSize+l end
testData.size = function() return teSize+lt end
end
-- classes: GLOBAL var!
classes = {'face','backg'}

-- Exports -------------------------------------------------------------------
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}
