require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'nnx'
require 'cunn'
require 'inn'

len1 = 12
len2 = 24
len3 =48

threshold48Calib = 0.1
scales = {0.07,0.10,0.05}

imgcnt = 1

inFolder = '../dataset/data/FaceLessImages/'
os.execute('mkdir AFLW_FaceLess_Patches')

function loadDataFiles(dir)
    local i, t, popen = 0, {}, io.popen  
    for filename in popen('ls -A "'..dir..'"' ):lines() do
	   i = i + 1
       t[i] = filename
    end
    return t
end

function cropImage(r,im,res)
	r = {x1 = torch.round(r.x1),y1 = torch.round(r.y1),x2 = torch.round(r.x2),y2 = torch.round(r.y2)}
	w = r.x2 - r.x1 + 1
	h = r.y2 - r.y1 + 1 
    local rc ={x1 = 1, y1 = 1, x2 = w , y2 = h}
    local imc=nil
       if(r.x1 < 1) then
            v = 1 - r.x1
            r.x1 = 1
            rc.x1 = v + 1
        end
        if(r.y1 < 1) then
            v = 1 - r.y1
            r.y1 = 1
            rc.y1 = v + 1
        end
        if(r.x2 > im:size(3)) then
            rc.x2 = w - r.x2 + im:size(3)
            r.x2 = im:size(3)
            
        end
        if(r.y2 > im:size(2)) then
            rc.y2 = h- r.y2 + im:size(2)
            r.y2 = im:size(2)
            
        end
    if(r.x1<r.x2 and r.y1<r.y2) then
        
        im = image.crop(im, r.x1, r.y1, r.x2,r.y2)
        imc = im:clone()
        if(rc.x1>1) then
            imc = torch.cat(torch.zeros(imc:size(1),imc:size(2),rc.x1-1),imc,3)
        end
        if(rc.x2<w) then
            imc = torch.cat(imc,torch.zeros(imc:size(1),imc:size(2),w - rc.x2),3)
        end
        if(rc.y1>1) then
            imc = torch.cat(torch.zeros(imc:size(1),rc.y1-1,imc:size(3)),imc,2)
        end
        if(rc.y2<h) then
            imc = torch.cat(imc,torch.zeros(imc:size(1),h - rc.y2,imc:size(3)),2)
        end
        imc = image.scale(imc,res,res)
        end
return imc            
end

torch.setdefaulttensortype('torch.FloatTensor')
network1 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/model.net')
net12 = nn.Sequential()
for i=1,5 do
	net12:add(network1.modules[i])
end
classifier1 = nn.Sequential()
for i=6,8 do 
	classifier1:add(network1.modules[i])
end
classifier = nn.SpatialClassifier(classifier1)

net12:add(classifier)
net12:float()

--calib12 = torch.load('model3Cascade/model12-calib.net') 
--calib12:float()

net24 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24net/results/model.net')--
net24:float()

calib24= torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24calibnet/results/model.net')
calib24:float()

net48 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48net/results/model.net')
net48:float()

calib48= torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48calibnetgray/results/model.net')
calib48:float()
-------------------------LabelTransformation---------------
local s = {0.83,0.91,1.0,1.10,1.21}
local xy = {-0.17,0,0.17}
T = torch.Tensor(45,3)
local cnt = 0
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
-------------------------------ENDLBLTRSFRM--------------------------

-- setup camera
--camera = image.Camera(0)

--scales = {1}
-- use a pyramid packer/unpacker
require 'PyramidPacker'
require 'PyramidUnPacker'
packer = nn.PyramidPacker(net12, scales)
unpacker = nn.PyramidUnPacker(net12)
-----------------PARSEFFI-------------------------------
function parseFFI(pin, threshold, blobs, scale)
  --loop over pixels
  for y=1, pin:size(2) do
     for x=1, pin:size(3) do
		a = math.exp(pin[1][y][x])
        if (a > threshold) then
          entry = {}
          entry[1] = x-1
          entry[2] = y-1
          entry[3] = scale
          entry[4] = a
          table.insert(blobs,entry)
      end
    end
  end
end
---------------------------------------------------------------------------
---------------------------------------------FORWARD CALIBNET--------------------------
function applyCalibNet(model,iim,detections,indeces,isize,threshold)
   
   if(indeces:numel()~=0) then
   local iimVec = torch.Tensor(indeces:size(1),1,isize,isize)
   for i=1,indeces:size(1) do
		detect=detections[indeces[i]]
		--local  imtmp = cropImage({x=detect[1], y=detect[2], w=detect[3]-detect[1], h=detect[4]-detect[2]},im,secondFilter)
		--im2[i] = imtmp:clone()
		--local imtmp = image.crop(im,detect[1], detect[2], detect[3],detect[4])
		--im2[i] = image.scale(imtmp,secondFilter,secondFilter)
		iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iim, isize)
		
	end
	--image.display{image=im2, nrow=32}
   
   --ca1 = torch.exp(calib12:forward(imVec:cuda()):cuda())
   local ca1 = torch.exp(model:forward(iimVec))
   --print(out1)
   local trans = torch.Tensor(ca1:size(1),3):zero()
   if average==1 then
		trans:addmm(ca1,T)
   else
		for i=1,ca1:size(1) do
			c = 0
			for j=1,ca1:size(2) do
				if(ca1[i][j] > threshold) then
					trans[i] = trans[i] + T[j]
					c = c + 1
				end
			end
			if (c ~= 0) then
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
			else
				trans[i][1] = 1
				trans[i][2] = 0
				trans[i][3] = 0
			end
		end
	end
   
   
   for i=1,trans:size(1) do
		--print(detections2[i][1]..' '..detections2[i][2]..' '..detections2[i][3]..' '.. detections2[i][4]..' '..im:size(2)..' '..im:size(3))
		w = detections[indeces[i]][3]-detections[indeces[i]][1]
		h = detections[indeces[i]][4]-detections[indeces[i]][2]
		sn = trans[i][1]
		xn = trans[i][2]
		yn = trans[i][3]
		
		--print(xn..' '..yn..' '..sn)
        x1 = torch.round(detections[indeces[i]][1] - xn*w/sn)
        y1 = torch.round(detections[indeces[i]][2] - yn*h/sn)
		x2 = torch.round(x1 + w/sn)
        y2 = torch.round(y1 + h/sn)
        
        detections[indeces[i]][1] = x1
        detections[indeces[i]][2] = y1
        detections[indeces[i]][3] = x2
        detections[indeces[i]][4] = y2
         
   end
   end
   return detections
end
-------------------------------END_APPLY_CALIBNET------------------------------------------------------
-------------------------------FORWARD_CALIBNET_CROPPED------------------------------------------------------
function applyCalibNetCropped(model,iims,detections,o,indeces,indecesNMS,isize,threshold,un,unT)
   if(#indeces~=0) then
   local iimVec = torch.Tensor(#indeces,1,isize,isize)
   for i=1,#indeces do
		iimVec[i] = iims[indeces[i]]:clone()
   end
	--image.display{image=im3, nrow=32}
   
   
   local ca = torch.exp(model:forward(iimVec))
   
   local trans = torch.Tensor(ca:size(1),3):zero()
   if average==1 then
		trans:addmm(ca,T)
   else
		for i=1,ca:size(1) do
			c = 0
			for j=1,ca:size(2) do
				if(ca[i][j] > threshold) then
					trans[i] = trans[i] + T[j]
					c = c + 1
				end
			end
			if(c ~= 0) then
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
			else
				trans[i][1] = 1
				trans[i][2] = 0
				trans[i][3] = 0
			end
		end
	end
   
   local detectionsR = torch.Tensor(#indeces,5)
   for i=1,#indeces do
		--print(detections2[i][1]..' '..detections2[i][2]..' '..detections2[i][3]..' '.. detections2[i][4]..' '..im:size(2)..' '..im:size(3))
		w = detections[indecesNMS[indeces[i]]][3]-detections[indecesNMS[indeces[i]]][1]
		h = detections[indecesNMS[indeces[i]]][4]-detections[indecesNMS[indeces[i]]][2]
		sn = trans[i][1]
		xn = trans[i][2]
		yn = trans[i][3]

		--print(xn..' '..yn..' '..sn)
		x1 = torch.round(detections[indecesNMS[indeces[i]]][1] - xn*w/sn)
        y1 = torch.round(detections[indecesNMS[indeces[i]]][2] - yn*h/sn)
		x2 = torch.round(x1 + w/sn)
        y2 = torch.round(y1 + h/sn)
        
        detectionsR[i][1] = x1
        detectionsR[i][2] = y1
		detectionsR[i][3] = x2
        detectionsR[i][4] = y2
        detectionsR[i][5] = o[indeces[i]][1]
        --print('edited '..detections2[i][1]..' '..detections2[i][2]..' '..detections2[i][3]..' '.. detections2[i][4]..' '..im:size(2)..' '..im:size(3))
   end
   if(un==1) then
		nmsKeep = nms(detectionsR,unT)
	else
		nmsKeep = torch.LongTensor(detectionsR:size(1))
		for j=1,nmsKeep:size(1) do
			nmsKeep[j] = j
		end
	end
   return detectionsR,nmsKeep
   else 
   return nil,nil
   end
   	
   --print(keep2nms:size(1))
end
--------------------------------------END_APPLY_CALIBNET_CROPPED----------------------------------------------------------------
--------------------------------------FORWARD_APPLY_NET---------------------------------------------------------
function applyNet(model,image,detections,indeces,isize,threshold)



   if(indeces:numel()~=0) then
   local iimVec = torch.Tensor(indeces:size(1),1,isize,isize)
   local k ={}
   for i=1,indeces:size(1) do
		detect=detections[indeces[i]]
		--local  imtmp = cropImage({x=detect[1], y=detect[2], w=detect[3]-detect[1], h=detect[4]-detect[2]},im,thirdFilter)
		--im2[i] = imtmp:clone()
		--local imtmp = image.crop(im,detect[1], detect[2], detect[3],detect[4])
		--im2[i] = image.scale(imtmp,thirdFilter,thirdFilter)
		iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, image, isize)
	end
	--image.display{image=imVec, nrow=32}
   
   --image.display{image=imVec, nrow=32}
   o = torch.exp(model:forward(iimVec))	
  local cnt = 1
   
   for i=1,o:size(1) do
       a = o[i][1]
       --b = out2[i][2]
	   if (a > threshold) then
			k[cnt] = i
			cnt = cnt +1
	   end	
   end
   return o,k,iimVec
   else
   return nil,nil,nil
   end
end

---------------------------END_APPLY_NET-------------------------------------
----------------------------------------------------NMS-----------------
function nms(boxes, overlap)
  
  local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  local x1 = boxes[{{},1}]
  local y1 = boxes[{{},2}]
  local x2 = boxes[{{},3}]
  local y2 = boxes[{{},4}]
  local s = boxes[{{},5}]
  
  local area = boxes.new():resizeAs(s):zero()
  area:map2(x2,x1,function(xx,xx2,xx1) return xx2-xx1+1 end)
  area:map2(y2,y1,function(xx,xx2,xx1) return xx*(xx2-xx1+1) end)

  local vals, I = s:sort(1)

  pick:resize(s:size()):zero()
  local counter = 1
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w = boxes.new()
  local h = boxes.new()

  while I:numel()>0 do 
    local last = I:size(1)
    local i = I[last]
    pick[counter] = i
    counter = counter + 1
    if last == 1 then
      break
    end
    I = I[{{1,last-1}}]
    
    xx1:index(x1,1,I)
    xx1:clamp(0,x1[i])
    yy1:index(y1,1,I)
    yy1:clamp(0,y1[i])
    xx2:index(x2,1,I)
    xx2:clamp(x2[i],math.huge)
    yy2:index(y2,1,I)
    yy2:clamp(y2[i],math.huge)
    
    w:resizeAs(xx2):zero()
    w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
    h:resizeAs(yy2):zero()
    h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
    
    local inter = w
    inter:cmul(h)
			
    local o = h
    xx1:index(area,1,I)
    
    o = torch.cdiv(torch.ones(inter:size())*area[i],inter)
    I = I[o:le(overlap)]
  end

  pick = pick[{{1,counter-1}}]
  return pick
end
---------------------------END_NMS----------------------------------------------------------------
network_fov = 12--12
network_sub = 4
----------------------------------PROCESS---------------------------------------------------------------------------------------
function process(imAdd)
   -- grab frame
   
   --frame = camera:forward()
   --frame = image.load('18.jpg')

   print(imAdd)
   local frame = image.load(inFolder..imAdd)
   print(frame:size())
if frame:size(2)<100 and frame:size(3)<100 then
return
end
   -- process input at multiple scales
   if (frame:size(1) == 1) then
	 local im1 = torch.FloatTensor(3,frame:size(2),frame:size(3))
	 im1[1] = frame:clone()  
	 im1[2] = frame:clone()  
	 im1[3] = frame:clone()
	 frame = im1;
   end
   imGray = image.rgb2y(frame):float() --
   local fmean = imGray:mean()
   local fstd = imGray:std()
   imGray:add(-fmean)
   imGray:div(fstd)  
   pyramid, coordinates, nScales = packer:forward(imGray)	
  
   det = net12:forward(pyramid)



   distributions = unpacker:forward(det, coordinates)
   threshold12 = .0
   threshold24 = .7
   threshold48 = .5
   threshold12Calib = .25
   threshold24Calib =.25
   local rawresults = {}

   -- function FFI:
   for i,distribution in ipairs(distributions) do
       parseFFI(distribution, threshold12, rawresults, nScales[i])
   end
   -- (7) clean up results
  
   local detections1 = torch.Tensor(#rawresults,5)
   for i,res in ipairs(rawresults) do
      local scale = res[3]
      local x = res[1]*network_sub/scale
      local y = res[2]*network_sub/scale
      local w = network_fov/scale
      local h = network_fov/scale
      detections1[{i,1}] = x
	  detections1[{i,2}] = y
	  detections1[{i,3}] = x+w
	  detections1[{i,4}] = y+h
	  detections1[{i,5}] = res[4]
   end
--print(#rawresults)   
if(#rawresults <1) then
	  return	
   end
   if(useNMS==1) then
		keep1nms = nms(detections1,0.8)
	else
		keep1nms = torch.LongTensor(detections1:size(1))
		for j=1,keep1nms:size(1) do
			keep1nms[j] = j
		end
	end
	--print("net12 done")
   --================Calibration 24 Net===================
  
   detections1 = applyCalibNet(calib24,imGray,detections1,keep1nms,len2,threshold12Calib)
   --print("net24-calib done")

   --======================24 Net=========================
  out2,keep2,imVec = applyNet(net24,imGray,detections1,keep1nms,len2,threshold24)
   --print("net24 done")   
--======================= SAVE FILE  ===================

       for i=1,keep1nms:size(1) do
		detect=detections1[keep1nms[i]]
		im2 = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, frame, len3)----------
		image.save('NegativeData2/'..string.sub(imAdd,1,string.find(imAdd,'.jpg')-1)..'_'..i..'.jpg',im2)
		i =i+1
print(i)
	end
--Offs=i+Offs



collectgarbage()
end
 ----------------------------------------------------------------------------------END_PROCESS---------------------------------------

address = loadDataFiles(inFolder)
for i=1,#address do
	process(address[i])
end
