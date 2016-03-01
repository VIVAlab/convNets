-- Config
PATH_TO_IMAGES = '/media/sf_ImageData/'
PATH_TO_POS = PATH_TO_IMAGES..'static/POS_DATA/'
PATH_TO_NEG = PATH_TO_IMAGES..'static/NEG_DATA/'
PATH_TO_GENERATED = PATH_TO_IMAGES.. 'generated/'

-- Evaluation Net Training Data --
--positiveData = {PATH_TO_IMAGES..'aflw/0/', PATH_TO_POS..'c_faces/', PATH_TO_POS..'faces/'}
aflwData = PATH_TO_IMAGES..'aflw/0/'
positiveData = {aflwData, PATH_TO_POS..'c_faces/', PATH_TO_POS..'faces/'}
negativeData = {PATH_TO_NEG..'negative/', PATH_TO_NEG..'nonfaces/', PATH_TO_NEG..'test_nonfaces/'}
-- lite data--
positiveLite = {PATH_TO_POS..'faces/'}
negativeLite = {PATH_TO_NEG..'negative/'}
-- Calibration Net Training Data --
calibDataRoot = '/media/sf_ImageData/'
calibDataCropped = calibDataRoot..'generated/croppednotdup/'
-- FALSE POSITIVES --
falsePositives = PATH_TO_GENERATED..'FaceLessImages/'
-- 24, 48 detection nets--
posTrain = {aflwData, PATH_TO_POS..'c_faces/'} 
posTest = {PATH_TO_POS..'faces/'}
negTrain = {PATH_TO_NEG..'negative/', PATH_TO_NEG..'nonfaces/', falsePositives}
negTest = {PATH_TO_NEG..'test_nonfaces/'}

-- MODEL LOCATIONS --
modelRoot = '../../GrayCascadeNet/'
---- Evaluation Nets ----
model12net = modelRoot..'12net/results/model.net'
model24net = modelRoot..'24net/results/model.net'
model48net = modelRoot..'48net/results/model.net'
---- Calibration Nets ----
model12calibnet = modelRoot..'12calibnet/results/model.net'
model24calibnet = modelRoot..'24calibnet/results/model.net'
model48calibnet = modelRoot..'48calibnet/results/model.net'

if(opt.size =='lite') then
   positiveData = positiveLite
   negativeData = negativeLite
   negTrain = {PATH_TO_GENERATED..'FaceLessImagesLite/'}
   posTrain = {PATH_TO_POS..'c_faces/'}
   calibDataCropped = calibDataRoot..'generated/croppednotdupLite/'
end

return {
   positiveData = positiveData,
   negativeData = negativeData,
   
   positiveTrain = posTrain,
   positiveTest = posTest,
   negativeTest = negTest,
   negativeTrain = negTrain,
   
   calibDataCropped = calibDataCropped,

   model12net = model12net,
   model24net = model24net,
   model48net = model48net,

   model12calibnet = model12calibnet,
   model24calibnet = model24calibnet,
   model48calibnet = model48calibnet
}
