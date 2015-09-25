require 'torch'
require 'image'
require 'xlua'
require 'torch'
require 'nnx'
require 'cunn'

secondFilter = 12
thirdFilter = 24
forthFilter = 48
threshold12 = 0.5
threshold24 = 0.5
threshold48 = 0.8
threshold12Calib = 0.05
threshold24Calib = 0.03
threshold48Calib = 0.02
average = 0
useNMS = 0
useGNMS = 1



scales = {0.6,0.4,0.3,0.25,0.2,0.16,0.13,0.1,0.08,0.07,0.06,0.05}

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
            --print('s0 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
            imc = torch.cat(torch.zeros(imc:size(1),imc:size(2),rc.x1-1),imc,3)
        end
        if(rc.x2<w) then
            --print('s1 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
            imc = torch.cat(imc,torch.zeros(imc:size(1),imc:size(2),w - rc.x2),3)
        end
        if(rc.y1>1) then
            --print('s2 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
            imc = torch.cat(torch.zeros(imc:size(1),rc.y1-1,imc:size(3)),imc,2)
        end
        if(rc.y2<h) then
            --print('s3 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
            imc = torch.cat(imc,torch.zeros(imc:size(1),h - rc.y2,imc:size(3)),2)
        
        end
        --print('st '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
        
        --if(imc:size(2)~=imc:size(3)) then
        --    print("problem in "..row.file_id.." and "..row.face_id)
        --end
        imc = image.scale(imc,res,res)
        end
return imc            
end


local base = '/home/mehdi/FDDB/FDDB-folds/'
local textFile = 'FDDB-'..folder..'.txt'
local imageFolder = '/home/mehdi/FDDB/facesInTheWild/'

local output = '/home/mehdi/FaceDetection2/model3Cascade/FDDB/'..folder..'-out.txt'
local file = io.open(base..textFile)
local outfile = io.open(output,'w+')

torch.setdefaulttensortype('torch.FloatTensor')
network1 = torch.load('model3Cascade/model12-net-v1.net')
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

calib12 = torch.load('model3Cascade/model12-calib.net')
calib12:float()

net24 = torch.load('model3Cascade/model24-net-v1.net')
net24.modules[1].modules[1].modules[1]:evaluate()
net24.modules[1].modules[1].modules[5]:evaluate()
net24.modules[1].modules[2].modules[2]:evaluate()
net24.modules[1].modules[2].modules[6]:evaluate()
net24.modules[3]:evaluate()
net24:float()



calib24= torch.load('model3Cascade/model24-calib.net')
calib24:float()

net48 = torch.load('model3Cascade/model48-net.net')
net48.modules[1].modules[1].modules[2]:evaluate()
net48.modules[1].modules[1].modules[6]:evaluate()
net48.modules[1].modules[2].modules[2]:evaluate()
net48.modules[1].modules[2].modules[6]:evaluate()
net48.modules[1].modules[3].modules[1]:evaluate()
net48.modules[1].modules[3].modules[5]:evaluate()
net48.modules[1].modules[3].modules[9]:evaluate()
net48.modules[3]:evaluate()
net48:float()

calib48= torch.load('model3Cascade/model48-calib.net')
calib48:float()

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


require 'PyramidPacker'
require 'PyramidUnPacker'
packer = nn.PyramidPacker(net12, scales)
unpacker = nn.PyramidUnPacker(net12)
function applyCalibNet(model,iim,detections,isize,threshold)
   
   if(detections~=nil) then
   local iimVec = torch.Tensor(detections:size(1),3,isize,isize)
   for i=1,detections:size(1) do
		detect=detections[i]
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
		w = detections[i][3]-detections[i][1]
		h = detections[i][4]-detections[i][2]
		sn = trans[i][1]
		xn = trans[i][2]
		yn = trans[i][3]
		
		--print(xn..' '..yn..' '..sn)
        x1 = torch.round(detections[i][1] - xn*w/sn)
        y1 = torch.round(detections[i][2] - yn*h/sn)
		x2 = torch.round(x1 + w/sn)
        y2 = torch.round(y1 + h/sn)
        
        detections[i][1] = x1
        detections[i][2] = y1
        detections[i][3] = x2
        detections[i][4] = y2
         
   end
   end
   return detections
end
function applyCalibNetCropped(model,iims,detections,o,isize,threshold,un,unT)
   if(detections~=nil) then   
   local ca = torch.exp(model:forward(iims))
   
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
   
   local detectionsR = detections
   for i=1,detections:size(1) do
		--print(detections2[i][1]..' '..detections2[i][2]..' '..detections2[i][3]..' '.. detections2[i][4]..' '..im:size(2)..' '..im:size(3))
		w = detections[i][3]-detections[i][1]
		h = detections[i][4]-detections[i][2]
		sn = trans[i][1]
		xn = trans[i][2]
		yn = trans[i][3]

		--print(xn..' '..yn..' '..sn)
		x1 = torch.round(detections[i][1] - xn*w/sn)
        y1 = torch.round(detections[i][2] - yn*h/sn)
		x2 = torch.round(x1 + w/sn)
        y2 = torch.round(y1 + h/sn)
        
        detectionsR[i][1] = x1
        detectionsR[i][2] = y1
		detectionsR[i][3] = x2
        detectionsR[i][4] = y2
        detectionsR[i][5] = o[i][1]
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
	
   local detectionsR1 = torch.Tensor()
   detectionsR1:index(detectionsR,1,nmsKeep)
   nmsKeep = nil
   return detectionsR1
   
   else 
   return nil
   end
 end

function applyNet(model,iimage,detections,isize,threshold)
   if(detections ~= nil) then
   local iimVec = torch.Tensor(detections:size(1),3,isize,isize)
   for i=1,detections:size(1) do
		detect=detections[i]
		iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iimage, isize)
	end
   o = torch.exp(model:forward(iimVec))	
  local cnt = 1
   k = torch.LongTensor(o:size(1))
   for i=1,o:size(1) do
       a = o[i][1]
       --b = out2[i][2]
	   if (a > threshold) then
			k[cnt] = i
			cnt = cnt +1
	   end	
   end
   if(cnt>1) then
   k = k[{{1,cnt-1}}]
   local o1 = torch.Tensor()
   o1:index(o,1,k)
   local iimVec1 = torch.Tensor()
   iimVec1:index(iimVec,1,k)
   local detectionsR = torch.Tensor()
   detectionsR:index(detections,1,k)
   return detectionsR,o1,iimVec1
   else
   return nil,nil,nil
   end
   else
   return nil,nil,nil
   end
end

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

network_fov = 12
network_sub = 2

if file then
    for line in file:lines() do
    detected = 0 
        print(imageFolder..line..'.jpg')
        im = image.load(imageFolder..line..'.jpg')
        if (im:size(1) == 1) then
			local im1 = torch.FloatTensor(3,im:size(2),im:size(3))
			im1[1] = im:clone()  
			im1[2] = im:clone()  
			im1[3] = im:clone()
			im = im1;
		end
		im = image.rgb2yuv(im):float()
		local fmean = im:mean()
		local fstd = im:std()
		im:add(-fmean)
		im:div(fstd)
        pyramid, coordinates, nScales = packer:forward(im)	
 
   det = net12:forward(pyramid)
   
   distributions = unpacker:forward(det, coordinates)
     
   rawresults = {}
   for i,distribution in ipairs(distributions) do
       parseFFI(distribution, threshold12, rawresults, nScales[i])
   end
   if(#rawresults <1) then
	  return	
   end
   -- (7) clean up results
   detections1 = torch.Tensor(#rawresults,5)
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
   if(useNMS==1) then
		keep1nms = nms(detections1,0.8)
	else
		keep1nms = torch.LongTensor(detections1:size(1))
		for j=1,keep1nms:size(1) do
			keep1nms[j] = j
		end
	end
	print("net12 done")
   --================Calibration 12 Net===================
   local ndetections1 = torch.Tensor()
   ndetections1:index(detections1,1,keep1nms)
   detections1 = ndetections1
   ndetections1 = nil
   detections1 = applyCalibNet(calib12,im,detections1,secondFilter,threshold12Calib)
   print("net12-calib done")
   --======================24 Net=========================
   detectionsT,out2,imVec = applyNet(net24,im,detections1,thirdFilter,threshold24)
   print("net24 done")
   --================Calibration 24 Net===================
   detections2 = applyCalibNetCropped(calib24,imVec,detectionsT,out2,thirdFilter,threshold24Calib,useNMS,0.7)
   print("net24-calib done")
   --print(keep2nms:size(1))
   --====================48 Filter========================	
   detectionsT,out3,imVec = applyNet(net48,im,detections2,forthFilter,threshold48)
   print("net48 done")
   --================Calibration 48 Net===================
   detections3 = applyCalibNetCropped(calib48,imVec,detectionsT,out3,forthFilter,threshold48Calib,useGNMS,0.3) 
   print("net48-calib done")
   
   
   if(detections3 ~= nil) then
        outfile:write(line)
        outfile:write('\n')
        outfile:write(detections3:size(1))
        outfile:write('\n')
       for i=1,detections3:size(1) do
        	detect=detections3[i]   
			outfile:write(detect[1]..' '..detect[2]..' '..detect[3]-detect[1]..' '..detect[4]-detect[2]..' '..detect[5]..'\n')
        end
    else
        outfile:write(line)
        outfile:write('\n')
        outfile:write('0\n')
		logFile:write(imageFolder..line..'.jpg')
		logFile:write('\n')
		logFile:flush()
	end
    end
else
end
outfile:close()
