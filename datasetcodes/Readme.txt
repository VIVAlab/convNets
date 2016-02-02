THE FOLDERS

1) The "cropped" folder contains all the calibration net training set. It also
contains images that where flipped left<->right for data augmentation. The fliped images are relabeled
since they can not possibly have the same label as the pre-flipped image.
Every example image is in a folder with it's corresponding label. ex label 1 image is in folder "1"

2) The "croppednotdup" is identical in structure to cropped folder exept that it does not contain flipped images.

THE SCRIPTS

1) "readData.lua" generates cropped+croppednotdup
2)"readDataFaceRemoval.lua" removes labelled faces from AFLW.
3) inside "croppednotdup" and "cropped" you will find "conversion.sh" run it after "readData.lua" is executed. Then remove .png files using the terminal with a command of the type : "find /target/directory -name '*.png' -type f -delete".

