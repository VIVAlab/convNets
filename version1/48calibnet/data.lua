--require('mobdebug').start()
require 'image'   -- to visualize the dataset
require 'math'
----------------------------------------------------------------------
local p=.9-- the (# training data)/(#test+training data) ratio
-- TODO insert a cycling function to move the training data set through the whole data set. For metaparameter optimisation.
local sizelim=17300; -- size limit upon training AND test set. 
		--that is p*sizelim~~number of trainingdata
		-- and(1-p)*sizelim~~number of testdata 
function loadDataFiles(face_dir)
    local i,train,test,popen = 0, {},{}, io.popen  
    for filename in popen('ls -A "'..face_dir..'"' ):lines() do --count number of pics in file
	   i = i + 1
	if i>sizelim then
		break
	end
    end 
	local totdata=math.floor(i*p)/p --total data amount usable for p to work
	local traindatanum=math.floor(i*p) --training data number
	local testdatanum=totdata-traindatanum --test data number
	local i,popen = 0, io.popen --reset count
 
    for filename in popen('ls -A "'..face_dir..'"' ):lines() do --read files
	i=i+1
	if i<=totdata then
		if i<=traindatanum then
		train[i] = face_dir..filename
		elseif (i>traindatanum) then
		test[i-traindatanum] = face_dir..filename

		end
	else 
		break
	end
    end
	
    return train,test,traindatanum,testdatanum
end

-----------------------------------------------------------------------


local desImaX = 48  --Image Width
local desImaY = 48  --Image Height
local ivch = 3
local numblbls=45 -- TODO eventually put a function that counts the number of folders to make the numblbls reading automatic
local crdnlty=torch.Tensor(numblbls,2)-- to store training data lengths and testdata lengths for each label respectively [#trainData_i,#testData_i],i in [1,45]

---------loop to load ALL data
for lbl=1,numblbls do --labels
imageslist, imageslistt, crdnlty[{lbl,1}], crdnlty[{lbl,2}] = loadDataFiles('/home/jblan016/FaceDetection/dataset/data/cropped/'..lbl..'/')
--imagelistt is wrong? should be last 24384 files: Answer NO its ok,imagelistt is the last half of files for the popen ORDER which is NOT linearly continuous
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
			trdata[j] = image.rgb2yuv(im)
		end
		imageslist = nil
   		print('train data loaded for label '..lbl)
	
		for j,filename in ipairs(imageslistt) do
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			tedata[j] = image.rgb2yuv(im)
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
			trdata[j+trsize-crdnlty[{lbl,1}]] = image.rgb2yuv(im)
			
		end
		imageslist = nil
   		print('train data loaded for label '..lbl)

	
		for j,filename in ipairs(imageslistt) do
			print(filename)
			local im =  image.load(filename):float()
			im =  image.scale(im,desImaX,desImaY)
			tedata[j+tesize-crdnlty[{lbl,2}]] = image.rgb2yuv(im)
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
