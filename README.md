# convNets
implementation of CNNs for face detection

FDDB folder:
- Before running FDDB the addresses shoud be set in runEvaluate.pl, runevaluate.lua and evaluate.lua
- evaluate.lua, PyramidPacker.lua and PyramidUnPacker.lua should be updated with final changes on the cascade classifier
- Use qlua runevaluate.lua to run FDDB

DEBUG folder
- Set addresses for running visual debugging tool
- Use qlua debugCascade.lua for running GUI

12net, 12calibnet, 24net, 24calibnet, 48net,48calibnet
- To modify size of data change the code in data.lua
- model.lua contains the architecture
- Use run.lua to run the training

* Jonathan has the latest architectures 

AFLWCropper folder
- readData.lua reads the faces from database and crop the faces and saves them


