require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'nnx'
require 'cunn'
require 'inn'

ich=1
FilterSize1 = 12
FilterSize2 = 24
FilterSize3 = 48
--=================Net thresholds
threshold12 = 0.5
threshold24 = 0.5
threshold48 = 0.5
threshold12Calib = 0.8
threshold24Calib = 0.8
threshold48Calib = 0.8
--=================NMS
nmsth1=.95
nmsth2=.95
gnmsth=.1   --unT
si=.99;
--=================Image Pyramid variables
MinFaceSize=40
xstride=2;
ystride=2;
scratio=1/math.sqrt(2)

imgcnt = 1
--== Transformation variables
T = torch.FloatTensor(45,3)
local cnt = 0
local s = {0.83,0.91,1.0,1.10,1.21}
local xy = {-0.17,0,0.17}
for i=1,5 do
	for j=1,3 do
		for k=1,3 do
		    cnt=cnt+1
			T[cnt][1] = s[i]
			T[cnt][2] = xy[j]
			T[cnt][3] = xy[k]
		end
	end
end

--=================LOAD DATA FILES ====
function loadDataFiles(dir)
    local t,i,popen = {},1,io.popen
    for filename in popen('ls -A "'..dir..'"' ):lines() do
       t[i] = dir..filename
       i=i+1
    end
    return t
end
--END_LOAD_DATA_FILES--





--=====================================LOADING SECTION
t = loadDataFiles('images/')

torch.setdefaulttensortype('torch.FloatTensor')
if ich==1 then
  stat12net={}
  stat12net.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/mean.dat')[1]
  stat12net.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/std.dat')[1]

  net12 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/model(var+flip).net')
  net12.modules[2]:evaluate()
  net12.modules[6]:evaluate()
  net12.modules[10]:evaluate()
  net12:float()


  stat12calib={}
  stat12calib.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/mean.dat')[1]
  stat12calib.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/std.dat')[1]
  
  calib12=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/model.net')  --best model is trained with face flipping and no dropout
  calib12:float()

  calib24 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24calibnet/results/model.net')
  calib24:float()
  --TEMPORARY STATS
  stat24calib={}
  stat24calib.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/mean.dat')[1]
  stat24calib.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/std.dat')[1]
  stat24net={}
  stat24net.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/mean.dat')[1]
  stat24net.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/std.dat')[1]
  stat48calib={}
  stat48calib.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/mean.dat')[1]
  stat48calib.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/std.dat')[1]
  stat48net={}
  stat48net.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/mean.dat')[1]
  stat48net.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/std.dat')[1]
  --END TEMPORARY STATS
  --[[
  stat24calib={}
  stat24calib.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24calibnet/results/mean.dat')[1]
  stat24calib.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24calibnet/results/std.dat')[1]
  --]]
  --[[
  stat24net={}
  stat24net.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24net/results/mean.dat')[1]
  stat24net.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24net/results/std.dat')[1]
  --]]
  net24 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24net/results/model.net') --==================
  net24:float()
  net24.modules[2]:evaluate()
  net24.modules[6]:evaluate()
  --[[
  stat48net={}
  stat48net.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48net/results/mean.dat')[1]
  stat48net.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48net/results/std.dat')[1]
  --]]
  net48 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48net/results/model.net')
  net48.modules[4]:evaluate()
  net48.modules[8]:evaluate()
  net48:float()

  calib48 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48calibnet/results/model.net')
  calib48:float()
  --[[
  stat48calibnet={}
  stat48calibnet.mean=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48calibnet/results/mean.dat')
  stat48calibnet.std=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48calibnet/results/std.dat')
  --]]
elseif ich==3 then
  error('NOT YET IMPLEMENTED FOR ICH==3')
end 
--========================================END LOAD===================--

-- setup GUI (external UI file)
widget = qtuiloader.load('g.ui')
win = qt.QtLuaPainter(widget.frame)


qt.connect(qt.QtLuaListener(widget.N),
              'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
              function (...)
                if imgcnt<#t then
                   imgcnt = imgcnt +1
                else
		  		  imgcnt = 1	
                end   
              end)

qt.connect(qt.QtLuaListener(widget.P),
              'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
              function (...)
                if imgcnt > 1  then
                 imgcnt = imgcnt - 1
                else
				 imgcnt = #t 
                end   
              end)
-- setup camera
--camera = image.Camera(0)


--=================================Net functions===

--=========================================APPLY_NET==================
function XNetBlock(model,modelstat,iim,detections,threshold,F1)  
     local iimVec=vectcrop(detections[{{},{1,4}}],iim,modelstat,F1)
     local probs = torch.exp(model:forward(iimVec))
     detections[{{},{5}}]=probs[{{},{1}}]:clone()
     local Mask=probs[{{},{1}}]:ge(threshold)
     if not torch.any(Mask) then
       return nil,0
     end
     probs=nil
     local vdetections=detections[torch.repeatTensor(Mask,1,5)]                                
     local detections=vdetections:resize(vdetections:size(1)/5,5)
     vdetections=nil
     return detections,detections:size(1)
end
--== END APPLY END==--

--=================CROPIMAGE ====
function cropImage(iarg,iim)
  --This function crops an image(CxHxW) with input arguments (x1,x2,y1,y2) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
  --The processing is done in matrix coordinates (i,j).
  local C=iim:size(1)
  local W=iim:size(3)
  local H=iim:size(2)
  
  local oim=torch.Tensor(C,iarg[4]-iarg[3]+1,iarg[2]-iarg[1]+1):fill(0)  --output image
  local Mask=torch.ByteTensor(C,iarg[4]-iarg[3]+1,iarg[2]-iarg[1]+1):fill(0)
  local iIxf=math.min(iarg[2],W)
  local iIx0=math.max(iarg[1],1)--1 because start index of image=1
  local DiIx=iIxf-iIx0+1 --span of elements in x copyied from input image to output image
  local Offsx=iIx0-math.min(iarg[1],iIx0) -- index offset of output image in x
  local iIyf=math.min(iarg[4],H)
  local iIy0=math.max(iarg[3],1)--1 because start index of image=1
  local DiIy=iIyf-iIy0+1 --span of elements in y copyied from input image to output image
  local Offsy=iIy0-math.min(iarg[3],iIy0) -- index offset of output image in y
 -- print(1,-1,iIy0,iIyf,iIx0,iIxf)
  local Ic=iim:sub(1,-1,iIy0,iIyf,iIx0,iIxf):clone()
  print(1,-1,Offsy+1,Offsy+DiIy,Offsx+1,Offsx+DiIx)
  print(Mask)
  Mask:sub(1,-1,Offsy+1,Offsy+DiIy,Offsx+1,Offsx+DiIx):fill(1)
  print(Mask)
  oim:maskedCopy(Mask,Ic) 
  --oim = image.scale(oim,res,res)
  --oim = image.rgb2y(oim)  --TODO change for grayscale option if C==3 consider cases where C=1 and C =4.
  return oim			
end
--=================END CROPIMAGE ====

--============================================NMS===========================
function nms(boxes, overlap)
  local Mask=torch.ByteTensor()
  local Dets = boxes[{{},{1,4}}]:clone()
  local s = boxes[{{},5}]:clone()
  local s, i = torch.sort(s,1,true)
  local N=Dets:size(1)
  for k=1,N do 
    Dets[k]=boxes[{i[k],{1,4}}]
  end
  Ivec=nil
  boxes=nil
  local Dtmp=torch.Tensor()
  local cnt=0
  i=torch.range(1,N,1):type('torch.LongTensor')
  while N>1 do
    cnt=cnt+1
    local IOU,IOUgt0=IoU(Dets[{cnt,{}}],Dets[{{cnt+1,-1},{}}])
    Mask=torch.ByteTensor(Dets[{{cnt+1,-1},{}}]:size(1)):fill(1)
    Mask[IOUgt0]=IOU[IOUgt0]:le(overlap)
    N=torch.sum(Mask)
    if N==0 then
      s=s[{{1,cnt}}]
      i=i[{{1,cnt}}]
      N=0
      Dets=Dets[{{1,cnt},{}}]
    else
      s=torch.cat(s[{{1,cnt}}],s[{{cnt+1,-1}}][Mask],1)
      i=torch.cat(i[{{1,cnt}}],i[{{cnt+1,-1}}][Mask],1)
      Mask=torch.repeatTensor(Mask,4,1):t()
      Dets=torch.cat(Dets[{{1,cnt},{}}],Dets[{{cnt+1,-1},{}}][Mask]:resize(N,4),1)
    end
  end
  return torch.cat(Dets,s,2),Dets:size(1)
end
--==END NMS==--
--=====================================GLOBAL_NMS===========================
function gnms(boxes, overlap,Ivec)
  local Dets = boxes[{{},{1,4}}]:clone()
  local s = boxes[{{},5}]:clone()
  local s, i = torch.sort(s,1,true) 
  local N=Dets:size(1)
  local Ivect=torch.Tensor(N,Ivec:size(2),Ivec:size(3),Ivec:size(4))
  for k=1,N do 
    Dets[k]=boxes[{i[k],{1,4}}]
    Ivect[k]=Ivec[i[k]]
  end
  Ivec=nil
  boxes=nil
  local Dtmp=torch.Tensor()
  local cnt=0
  i=torch.range(1,N,1):type('torch.LongTensor')
  while N>1 do
    cnt=cnt+1
    local IOU,IOUgt0=IoU(Dets[{cnt,{}}],Dets[{{cnt+1,-1},{}}])
    Mask=torch.ByteTensor(Dets[{{cnt+1,-1},{}}]:size(1)):fill(1)
    Mask[IOUgt0]=IOU[IOUgt0]:le(overlap)
    N=torch.sum(Mask)
    if N==0 then
      s=s[{{1,cnt}}]
      i=i[{{1,cnt}}]
      N=0
      Dets=Dets[{{1,cnt},{}}]
    else
      s=torch.cat(s[{{1,cnt}}],s[{{cnt+1,-1}}][Mask],1)
      i=torch.cat(i[{{1,cnt}}],i[{{cnt+1,-1}}][Mask],1)
      Mask=torch.repeatTensor(Mask,4,1):t()
      Dets=torch.cat(Dets[{{1,cnt},{}}],Dets[{{cnt+1,-1},{}}][Mask]:resize(N,4),1)
    end
  end
    OutIVect=torch.Tensor(s:size(1),Ivect:size(2),Ivect:size(3),Ivect:size(4))
    for k=1,s:size(1) do
       OutIVect[k]=Ivect[i[k]]
    end
  Ivect=nil
  return torch.cat(Dets,s,2),Dets:size(1),OutIVect
end
--==END GLOBAL NMS==--
--IoU--====================
function IoU(d,D)
  local N=D:size(1)
  local InoU=torch.Tensor(N):fill(0) --intersection for now
  local O=torch.Tensor(N):fill(0) --overlap
  local logica=D[{{},2}]:lt(d[1]) --x2'<x1
  local logicb=D[{{},1}]:gt(d[2]) --x1'<x2
  local Xfit=(logica+logicb):ge(1)  --logic OR. if X fits, Xfit=1 else Xfit=0
  logica=nil
  logicb=nil
  local logicc=D[{{},4}]:lt(d[3]) --y2'<y1
  local logicd=D[{{},3}]:gt(d[4]) --y1'<y2
  local Yfit=(logicc+logicd):ge(1)  --logic OR. if Y fits, Yfit=1 else Yfit=0
  logicc=nil
  logicd=nil
  local XYfit=Xfit:eq(Yfit)  --XYfit =(Xfit AND Yfit)
  --compute intersectionarea
  InoU[XYfit]=torch.cmul(torch.clamp(D[{{},2}][XYfit],d[1],d[2])-torch.clamp(D[{{},1}][XYfit],d[1],d[2])+1,torch.clamp(D[{{},4}][XYfit],d[3],d[4])-torch.clamp(D[{{},3}][XYfit],d[3],d[4])+1) 
  --compute overlap
  O[XYfit]=torch.cmul(D[{{},2}][XYfit]-D[{{},1}][XYfit]+1,D[{{},4}][XYfit]-D[{{},3}][XYfit]+1)+math.pow(d[2]-d[1]+1,2)-InoU[XYfit] --compute only overlaps for non-zero intersections
  InoU[XYfit]=torch.cdiv(InoU[XYfit],O[XYfit])
  return InoU, XYfit
end

--END_IoU=======================-


--==APPLY CALIBRATION NET=====================================================================-----
function XCalibNetBloc(model,modelstat,iim,detections,F1,F2,threshold)
     --print(detections)
     --local detections = coordinatescaling(F1,F2,detections)
     --print(detections)
     --error('287')
     local iimVec=vectcrop(detections[{{},{1,4}}],iim,modelstat,F1)
     local probs = torch.exp(model:forward(iimVec))  -- probs_(#det x 45 )
     local trans = torch.Tensor(probs:size(1),3):zero()
     
     for i=1,probs:size(1) do-- for each detection window 
	   c = 0
	   local maxlbl=torch.max(probs[i])
       for j=1,probs:size(2) do -- for all 45 labels
	     if(probs[i][j] > threshold) then
	       trans[i] = trans[i] + T[j]
	        c = c + 1
	     end
       end
       if (c~=0) then --TODO pick a better way of choosing max
	     trans[i][1] = trans[i][1]/c
	     trans[i][2] = trans[i][2]/c
	     trans[i][3] = trans[i][3]/c
       elseif (c == 0) then
	     for j=1,probs:size(2) do
	       if(probs[i][j] == maxlbl) then
	         trans[i] = trans[i] + T[j]
	         c = c + 1
	       end
	     end
         trans[i][1] = trans[i][1]/c
         trans[i][2] = trans[i][2]/c
         trans[i][3] = trans[i][3]/c
       end
     end
     detections = coordinatescaling(F1,F2,detections)
   
     for i=1,trans:size(1) do 
        local w = math.abs(detections[i][2]-detections[i][1]+1)--
        local h = math.abs(detections[i][4]-detections[i][3]+1)--
        local sn = trans[i][1]
        local xn = trans[i][2]
        local yn = trans[i][3]
        detections[i][1] = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn) --x1
        detections[i][3] = torch.round(detections[i][3] -h*yn+(sn-1)*h/2/sn) --y1
        detections[i][2] = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn+w/sn) --x2
        detections[i][4] = torch.round(detections[i][3] -h*yn+(sn-1)*h/2/sn+h/sn) --y2
     end
   return detections
end

--===END_APPLY CALIBNET===--
-- MULTIRESOLUTION CALIBRATION NET ===
function XCalibNetBlocMultiresln(model,iimVec,detections,threshold)

     local probs = torch.exp(model:forward(iimVec))  -- probs_(#det x 45 )
     local trans = torch.Tensor(probs:size(1),3):zero()

     for i=1,probs:size(1) do-- for each detection window 
	c = 0
	local maxlbl=torch.max(probs[i])
	for j=1,probs:size(2) do -- for all 45 labels
	  if(probs[i][j] > threshold) then
	    trans[i] = trans[i] + T[j]
	    c = c + 1
	  end
	end
	if (c~=0) then --TODO pick a better way of choosing max
	  trans[i][1] = trans[i][1]/c
	  trans[i][2] = trans[i][2]/c
	  trans[i][3] = trans[i][3]/c
	elseif (c == 0) then
	  for j=1,probs:size(2) do
	    if(probs[i][j] == maxlbl) then
	      trans[i] = trans[i] + T[j]
	      c = c + 1
	    end
	  end
	trans[i][1] = trans[i][1]/c --might be useless
	trans[i][2] = trans[i][2]/c  --might be useless
	trans[i][3] = trans[i][3]/c --might be useless
        end
      end

   detections = coordinatescaling(F1,F2,detections)
   
     for i=1,trans:size(1) do 
        local w = math.abs(detections[i][2]-detections[i][1]+1)--
	local h = math.abs(detections[i][4]-detections[i][3]+1)--
	local sn = trans[i][1]
	local xn = trans[i][2]
	local yn = trans[i][3]
	detections[i][1] = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn) --x1
        detections[i][3] = torch.round(detections[i][3] -h*yn+(sn-1)*h/2/sn) --y1
	detections[i][2] = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn+w/sn) --x2
        detections[i][4] = torch.round(detections[i][3] -h*yn+(sn-1)*h/2/sn+h/sn) --y2
     end
   return detections 
end
--END MULTIRESOLUTION CALIBREATION NET ==


function detcoords(filtsize,dj,di,h,w)
  --takes for input the filter size, the strides +the image size --TODO change this to detection windows as if image was full and filtsize=MinfaceSize/s full image
  --outputs detection boxes to scan an image fully. The input filtersize and strides are already scaled.
  --outputs detections matrix and last column reserved for scores :D_{Ni*Nj x 5} Matrix and Ni*Nj
  
  local ri=((h-filtsize)%di)/di
  local rj=((w-filtsize)%dj)/dj
  local iOffset=-di*ri
  local jOffset=-dj*rj

   local Ni=math.floor((h-filtsize+2*ri)/di)+1 --(padding of ri in the left of index i)

   local Nj=math.floor((w-filtsize+2*rj)/dj)+1


  local Detections=torch.Tensor(Nj*Ni,5):fill(0)
  Detections[{{},1}]= torch.repeatTensor(torch.range(iOffset+1,di*(Ni-1)+iOffset+1,di),1,Nj)--x1s
  Detections[{{},2}]=Detections[{{},1}]+filtsize-1  --x2s
  Detections[{{},3}]=torch.repeatTensor(torch.range(jOffset+1,dj*(Nj-1)+jOffset+1,dj),1,Ni)  --y1s
  Detections[{{},4}]=Detections[{{},3}]+filtsize-1  --y2s
  --Detections[{{},5}] :reserved for net outputs
return Detections, Ni*Nj 
end 

function coordinatescaling(F1,F2,Detections)
  --Rescales the F1xF1 detections so that a F2xF2 filter covers Fs x Fs windows. The output is the D_{#Detsx5} matrix (last column reserved for scores)
  local a = (F2-1)/(F1-1)
  Detections[{{},{1,4}}] = torch.add(torch.round(torch.mul(torch.add(Detections[{{},{1,4}}],-1),a)),1)
  return Detections  
end

function pyramidscales(FilterSize,MinFaceSize,sc,imW,imH)  --can fuse this with scan
  --Outputs the number of pyramid levels and the scaling factors given (FilterSize,Minimum Face Size,scaling factor step,image Height,image Width)   
  --scaling factor  0<sc<1  
  local Npl=math.floor(-math.log(math.min(imW,imH)/MinFaceSize)/math.log(sc))+1 --MAx number of pyramid levels given image width height, minimum face size and scaling level ratio (smaller than 1)
  local scales={}

  local s=FilterSize/MinFaceSize
Fs=torch.Tensor(Npl)
  for k=1,Npl do
    scales[k]=s*(sc)^(Npl-k)
    Fs[k]=MinFaceSize/(sc)^(Npl-k)-- Apparent filtersize
  end
  return scales, Npl
end

function returnimage(frm)
  local W=frm:size(3)
  local H=frm:size(2)
  local frame2image=torch.Tensor()
  if (frame:size(1) == 1) then
    if ich==1 then
      local frame2image = frm:clone():float()
    elseif ich==3 then
      local frame2image = torch.Tensor(ich,frm:size(2),frm:size(3))
      for k=1,ich do
        frame2image[k]=frm:clone():float()
      end
    end
  elseif (frm:size(1) == 3) then
    if ich==1 then
      frame2image = image.rgb2y(frm):float()
    elseif ich==3 then
      frame2image = frm:clone():float()
    end
  end
  return frame2image,W,H
end
--==VECTCROP======-
function vectcrop(iarg,iim,stats,F)
--Returns a Tensor of normalised patches Ivect_{batch#,ichs,H,W} given the inputs: statistics list,the scaled input image, and the Detections only ( no score column in j=5)
 iim:add(-stats.mean)
 iim:div(stats.std)
 local B=iarg:size(1) --batch size
 I=torch.Tensor()
 if iarg[1][2]-iarg[1][1]+1~=F then
   I=image.scale(cropImage(iarg[1],iim),F,F) --return first cropped image
 else
   I=cropImage(iarg[1],iim)
 end
 
 local ichs=I:size(1)
 local H=I:size(2)
 local W=I:size(3)
 local Ivect=torch.Tensor(B,ichs,H,W)
 
 Ivect[1]=torch.reshape(I,1,ichs,H,W) --Maybe unnecessary
 if B>1 then 
   for id=2,B do
     if iarg[id][2]-iarg[id][1]+1~=F then
       I=image.scale(cropImage(iarg[id],iim),F,F)
     else
     print('IARG1',iarg[1])
     print(iarg[id][1],iarg[id][2],iarg[id][3],iarg[id][4],'H='..iim:size(2),'W='..iim:size(3))
     
       I=cropImage(iarg[id],iim)
     end
     Ivect[id]=I
   end
 end
 return Ivect
end
--=====END_VECTCROP ======-
-------------------------======================PROCESS============----------
function process()
   -- grab frame
   --frame = camera:forward()
   --frame = image.load('18.jpg')
   --WIDGET
     threshold12 = widget.s1.value/10000
     threshold12Calib = widget.s2.value/10000
     nmsth1=widget.s3.value/10000
     threshold24 =widget.s4.value/10000
     threshold24Calib = widget.s5.value/10000
     nmsth2=widget.s6.value/10000
     threshold48 = widget.s7.value/10000
     gnmsth=widget.s8.value/10000
     threshold48Calib = widget.s9.value/10000
     print('threshold12 = '..threshold12,' threshold12Calib = '..threshold12Calib,' nmsth1 = '..nmsth1)
     print('threshold24 = '..threshold24,' threshold24Calib = '..threshold24Calib,' nmsth2 = '..nmsth2)
     print('threshold48 = '..threshold48,' gnmsth = '..gnmsth,' threshold48Calib = '..threshold48Calib)
     --=================NMS

     
     

     frame = image.load(t[imgcnt])
     local img,W,H = returnimage(frame)
     local scales,Npl = pyramidscales(FilterSize1,MinFaceSize,scratio,W,H)
     local Detections=torch.Tensor()
     local Detections12net=torch.Tensor()
     local DetectionsNMS1=torch.Tensor()
     local Detections12cnet=torch.Tensor()
     local Detections24net=torch.Tensor()
     local DetectionsNMS2=torch.Tensor()
     local Detections24cnet=torch.Tensor()
     local Detections48net=torch.Tensor()
     local DetectionsGNMS=torch.Tensor()
     
     local IVECT=torch.Tensor()
     local NoDets=0
     local DetFlag=0;
   for Nsc=1,Npl do
     ::pyramidlevel::
     print('NSC========================================================'..Nsc,'Sc='..scales[Nsc])
     --preprocessing block

     local Fsc = FilterSize1/scales[Nsc]
     local xstridesc = xstride/scales[Nsc]
     local ystridesc = ystride/scales[Nsc]
     
     local Wsc = torch.round(W*scales[Nsc])
     local Hsc = torch.round(H*scales[Nsc])
     
     local imgsc = image.scale(img,Wsc,Hsc)  
     image.display(imgsc)

     --local detections,nodets=detcoords(Fsc,xstridesc,ystridesc,H,W)

     --local detections =coordinatescaling(Fsc,FilterSize1,detections)

     detections,nodets=detcoords(FilterSize1,xstride,ystride,Hsc,Wsc)
     --end preprocessing block

     

     print(detections)
     print(Hsc,Wsc)
     detections,nodets=XNetBlock(net12,stat12net,imgsc,detections,threshold12,FilterSize1)  
     
     if nodets==0 then
       if Nsc==Npl then
         if DetFlag==1 then
           goto endwithdetection
         elseif DetFlag==0 then
           goto nildetection
           ::nildetection::
           error('No detection whatsoever. Error for Demo only. For application, consider skip all.')
         end
       elseif Nsc<Npl then
              Nsc=Nsc+1
       goto pyramidlevel --skip a pyramid level
       end
     else
       DetFlag=1
     end
     local detections12net=detections
     
     print("net12 done")
     --================Calibration 12 Net===================
     detections = XCalibNetBloc(calib12,stat12calib,imgsc,detections,FilterSize1,FilterSize2,threshold12Calib)
     local detections12cnet=detections  
     print("net12-calib done") 
         detections,nodets = nms(detections,nmsth1)
     detectionsNMS1=detections
     print("nms1 done")
     --======================24 Net=========================
     imgsc = image.scale(img,2*Wsc,2*Hsc)
     detections,nodets=XNetBlock(net24,stat24net,imgsc,detections,threshold24,FilterSize2)  
     if nodets==0 then
       if Nsc==Npl then
         if DetFlag==1 then
           goto endwithdetection
         elseif DetFlag==0 then
           goto nildetection
           ::nildetection::
           error('No detection whatsoever. Error for Demo only. For application, consider skip all.')
         end
       elseif Nsc<Npl then
              Nsc=Nsc+1
       goto pyramidlevel --skip a pyramid level
       end
     else
       DetFlag=1
     end
     detections24net=detections
     print("net24 done")
     --================Calibration 24 Net===================
     detections = XCalibNetBloc(calib24,stat24calib,imgsc,detections,FilterSize2,FilterSize3,threshold24Calib)
     detections24cnet=detections
     print("net24-calib done") 
         detections,nodets = nms(detections,nmsth2)
     detectionsNMS2=detections
     print("nms2 done")
     
     --====================48 Filter========================	
     imgsc = image.scale(img,4*Wsc,4*Hsc)
     detections,nodets=XNetBlock(net48,stat48net,imgsc,detections,threshold48,FilterSize3)
     if nodets==0 then
       if Nsc==Npl then
         if DetFlag==1 then
           goto endwithdetection
         elseif DetFlag==0 then
           goto nildetection
           ::nildetection::
           error('No detection whatsoever. Error for Demo only. For application, consider skip all.')
         end
       elseif Nsc<Npl then
              Nsc=Nsc+1
       goto pyramidlevel --skip a pyramid level
       end
     else
       DetFlag=1
     end
     detections48net=detections
     print("net48 done")
      --CONCATENATION
    if Nsc==1 then  --TODO add other detections for visualisation
      Detections=detections
      IVECT=vectcrop(detections[{{},{1,4}}],imgsc,statcalib48)
      NoDets=nodets
      Detections12net=detections12net
      DetectionsNMS1=detectionsNMS1
      Detections12cnet=detections12cnet
      Detections24net=detections24net
      DetectionsNMS2=detectionsNMS2
      Detections24cnet=detections24cnet
      Detections48net=detections48net
    else
      IVECT=torch.cat(IVECT,vectcrop(detections[{{},{1,4}}],imgsc,statcalib48),1)
      Detections=torch.cat(Detections,detections,1)
      Detections12net=torch.cat(Detections12net,detections12net,1)
      DetectionsNMS1=torch.cat(DetectionsNMS1,detectionsNMS1,1)
      Detections12cnet=torch.cat(Detections12cnet,detections12cnet,1)
      Detections24net=torch.cat(Detections24net,detections24net,1)
      DetectionsNMS2=torch.cat(DetectionsNMS2,detectionsNMS2,1)
      Detections24cnet=torch.cat(Detections24cnet,detections24cnet,1)
      Detections48net=torch.cat(Detections48net,detections48net,1)
      NoDets=NoDets+nodets
    end
  end  --end loops of pyramid levels
     ::endwithdetection::
       Detections,NoDets,IVECT = gnms(Detections,gnmsth,IVECT)
     DetectionsGNMS=Detections
     print('gnms done')
   --================Calibration 48 Net===================
     Detections= XCalibNetBlocMultiresln(calib48,IVECT,threshold48Calib)
     print("net48-calib done")
     
end
  
-- display function TODO modify so that it shows every STEP
function display()
   zoom = 1
   win:gbegin()
   win:showpage()
   image.display{image=frame, win=win, zoom=zoom}
   if widget.t1.checked then
		t1 = 1
   else
		t1 = 0
   end
   if widget.t2.checked then
		t2 = 1
   else
		t2 = 0
   end
   if widget.t3.checked then
		t3 = 1
   else
		t3 = 0
   end
   if widget.t4.checked then
		t4 = 1
   else
		t4 = 0
   end
   if widget.t5.checked then
		t5 = 1
   else
		t5 = 0
   end
   if widget.t6.checked then
		t6 = 1
   else
		t6 = 0
   end
   if widget.t7.checked then
		t7 = 1
   else
		t7 = 0
   end
   if widget.t8.checked then
		t8 = 1
   else
		t8 = 0
   end
   if widget.t9.checked then
		t9 = 1
   else
		t9 = 0
   end
   if t1==1 then
       win:setcolor(.8,.1,.1)
       win:setlinewidth(1)	
       for i=1,Detections12net:size(1) do
          detect=Detections12net[i]:narrow(1,1,5)
          win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2]) --x1,y1,w,h
          win:stroke()
          win:setfont(qt.QFont{serif=false,italic=false,size=16})
          win:moveto(detect[1], detect[2]-1)
          win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t2==1 then
       win:setcolor(.1,.1,.8)
       win:setlinewidth(1)
       for i=1,DetectionsNMS1:size(1) do
          detect=DetectionsNMS1[i]
          win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
          win:stroke()
          win:setfont(qt.QFont{serif=false,italic=false,size=16})
          win:moveto(detect[1], detect[2]-1)
          win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t3==1 then
       win:setcolor(.1,.8,.1)
       win:setlinewidth(2)
       for i=1,Detections12cnet:size(1) do
         detect=Detections12cnet[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t4==1 then
       win:setcolor(.7,0,0)
       win:setlinewidth(2)
       for i=1,Detections24net:size(1) do
         detect=Detections24net[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t5==1 then
       win:setcolor(0,0,.7)
       win:setlinewidth(2)
       for i=1,DetectionsNMS2:size(1) do
         detect=DetectionsNMS2[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t6==1 then
       win:setcolor(0,.7,0)
       win:setlinewidth(2)
       for i=1,Detections24cnet:size(1) do
         detect=Detections24cnet[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t7==1 then
       win:setcolor(1,0,0)
       win:setlinewidth(2)
       for i=1,Detections48net:size(1) do
         detect=Detections48net[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t8==1 then
       win:setcolor(0,0,1)
       win:setlinewidth(2)
       for i=1,DetectionsGNMS:size(1) do
         detect=DetectionsGNMS[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
   if t9==1 then
       win:setcolor(0,1,0)
       win:setlinewidth(2)
       for i=1,Detections:size(1) do
         detect=Detections[i]
         win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
         win:stroke()
         win:setfont(qt.QFont{serif=false,italic=false,size=16})
         win:moveto(detect[1], detect[2]-1)
         win:show(string.format("%1.2f",detect[5]))
       end
   end
--	for i=1,#t do
--		win:setcolor(1,0,0)
--		win:rectangle(t[i][2]*network_sub, t[i][1]*network_sub, 64, 64)
--		win:stroke()
--      win:setfont(qt.QFont{serif=false,italic=false,size=16})
--      win:moveto(detect.x, detect.y-1)
--      win:show('face')
--	end
   win:gend()
   Detections12net=nil
   DetectionsNMS1=nil
   Detections12cnet=nil
   Detections24net=nil
   DetectionsNMS2=nil
   Detections24cnet=nil
   Detections48net=nil
   DetectionsGNMS=nil
   Detections=nil
end

-- setup gui

local timer = qt.QTimer()
timer.interval = 100
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              process()
              display()
              timer:start()
           end)
widget.windowTitle = 'A Simple Frame Grabber'
widget:show()
timer:start()
