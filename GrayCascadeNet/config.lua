-- Config
PATH_TO_IMAGES = '/home/joe/Documents/imageData/'

-- Training Data --
POSITIVE_TRAIN_FOLDERS = {PATH_TO_IMAGES..'TRAIN/positive/'}
NEGATIVE_TRAIN_FOLDERS = {PATH_TO_IMAGES..'TRAIN/negative/'}

-- Test Data --
POSITIVE_TEST_FOLDERS = {PATH_TO_IMAGES..'TEST/positive/'}
NEGATIVE_TEST_FOLDERS = {PATH_TO_IMAGES..'TEST/negative/'}

POSITIVE_DATA = {PATH_TO_IMAGES..'TRAIN/positive/', PATH_TO_IMAGES..'TEST/positive/'}
NEGATIVE_DATA = {PATH_TO_IMAGES..'TRAIN/negative/', PATH_TO_IMAGES..'TEST/negative/'}

GENERATED_NEGATIVES_FOLDER = {PATH_TO_IMAGES..'TRAIN/negative/'} -- should be false positives

positiveTrain = POSITIVE_TRAIN_FOLDERS
negativeTrain = NEGATIVE_TRAIN_FOLDERS

positiveTest = POSITIVE_TEST_FOLDERS
negativeTest = NEGATIVE_TEST_FOLDERS

positiveData = POSITIVE_DATA
negativeData = NEGATIVE_DATA

generatedNegativeData = GENERATED_NEGATIVES_FOLDER
return {
   positiveTrain = positiveTrain,
   negativeTrain = negativeTrain,
   positiveTest = positiveTest,
   negativeTest = negativeTest,
   positiveData = positiveData,
   negativeData = negativeData,
   generatedNegativeData = generatedNegativeData
}
