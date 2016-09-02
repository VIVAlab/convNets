There are three demos here: 'Demo.lua', 'mdemo3cuda.lua' and 'mdemocuda.lua'.

'mdemo3cuda.lua' runs on the gpu , works on multiple scales and uses three stages in the cascade(as in the paper).
Note that if you wish to skip any of the classification layers(face-detectors and calibration), set the threshold to 0. To skip nms, set the nms threshold to 1.

'mdemocuda.lua' runs on the gpu , works on multiple scales and uses only 2 stages in the cascade.


to execute the GUI, at the terminal type for example:qlua mdemo3cuda.lua
