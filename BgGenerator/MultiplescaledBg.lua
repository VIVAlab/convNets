require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'nnx'
require 'cunn'
require 'inn'

FilterSize1 = 12
FilterSize2 = 24
FilterSize3 = 48
threshold12 = 0.5 --.5
threshold24 = 0.0 --.5
threshold48 = 0.0
threshold12Calib = 0.5
threshold24Calib = 0.6
threshold48Calib = 0.7
nmsth1=.95
nmsth2=.95 --.7
gnmsth=.2   --unT
average = 0
useNMS = 1 --un 
useGNMS = 1
ich=1
si=.99
minF=40 --minimum Face size 40x40
Nscales=8
scratio=math.sqrt(2);
s=FilterSize1/minF
scales={}

for k=1,Nscales do
scales[k]=s*(1/scratio)^(k-1)
end



imgcnt = 1

inFolder = '../dataset/data/FaceLessImages/'--'../dataset/negative/'--
os.execute('mkdir AFLW_FACELESS_PATCHES')

function loadDataFiles(dir)
    local i, t, popen = 0, {}, io.popen  
    for filename in popen('ls -A "'..dir..'"' ):lines() do
	   i = i + 1
       t[i] = filename
    end
    return t
end
--======================bbox
function bbox(imgfr)
local flg=0
local gimg=image.rgb2y(imgfr)
local isblack=gimg:eq(0)
isblack:apply(function(isb) flg=flg+isb end)
return flg
end
--======================END_bbox
--=============================cropImageblackbox=====================
function cropImagebb(r,im,res)
	r = {x1 = torch.round(r.x1),y1 = torch.round(r.y1),x2 = torch.round(r.x2),y2 = torch.round(r.y2)}
	w = r.x2 - r.x1 + 1
	h = r.y2 - r.y1 + 1 
    local rc ={x1 = 1, y1 = 1, x2 = w , y2 = h}
    local imc=nil
    local bflg=0
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
	
        
    end
	bflg=bbox(imc)
	imc = image.scale(imc,res,res)
return imc,bflg           
end
--==========================================END_CROP_IMAGEblackbox===============-
--=============================cropImage=====================
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
        
    end
	imc = image.scale(imc,res,res)
	local imcm=imc:mean()
	local imcs=imc:std()
	imc:add(-imcm)
	imc:div(imcs)
return imc            
end
--==========================================END_CROP_IMAGE===============-
--=============================cropImageNonorm=====================
function cropImagenonorm(r,im,res)
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
        
    end
	imc = image.scale(imc,res,res)
return imc            
end
--==========================================END_CROP_IMAGENoNorm===============-
-----------------------load NETS

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

calib12=torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12calibnet/results/model.net')
calib12:float()
calib24 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24calibnet/results/model.net')
calib24:float()

net24 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/24net/results/model.net')
net24:float()


net48 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48net/results/model.net')
net48.modules[4]:evaluate()
net48.modules[8]:evaluate()
net48:float()

calib48 = torch.load('/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/48calibnet/results/model.net')
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
-------------------------------END_PARSEFFI---------------------------------------
--====================================================APPLY_CALIB_NET
function applyCalibNet(model,iim,detections,isize,threshold)
   
   if(detections~=nil) then
   local iimVec = torch.Tensor(detections:size(1),ich,isize,isize)
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
			local maxlbl=torch.max(ca1[i])
			for j=1,ca1:size(2) do
				if(ca1[i][j] > threshold) then
					trans[i] = trans[i] + T[j]
					c = c + 1
				end
				
			end
			if (c~=0) then
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
			elseif (c == 0) then
				for j=1,ca1:size(2) do
					if(ca1[i][j] == maxlbl) then
						trans[i] = trans[i] + T[j]
						c = c + 1
					end
				end
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
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
--=========================================END_APPLY_CALIB_NET
---========================================APPLY_CALIB_NET_CROPPED==========================
function applyCalibNetCropped(model,iims,detections,o,isize,threshold,un,unT,g)
   if(detections~=nil) then   
   local ca = torch.exp(model:forward(iims))
   
   local trans = torch.Tensor(ca:size(1),3):zero()
   if average==1 then
		trans:addmm(ca,T)
   else
		for i=1,ca1:size(1) do
			c = 0
			local maxlbl=torch.max(ca1[i])
			for j=1,ca1:size(2) do
				if(ca1[i][j] > threshold) then
					trans[i] = trans[i] + T[j]
					c = c + 1
				end
				
			end
			if (c~=0) then
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
			elseif (c == 0) then
				for j=1,ca1:size(2) do
					if(ca1[i][j] == maxlbl) then
						trans[i] = trans[i] + T[j]
						c = c + 1
					end
				end
				trans[i][1] = trans[i][1]/c
				trans[i][2] = trans[i][2]/c
				trans[i][3] = trans[i][3]/c
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
        if(g==0) then
		nmsKeep = nms(detectionsR,unT)--,si)
		else
		nmsKeep = gnms(detectionsR,unT)
		end
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
--==================================================END_APPLY_CALIBNET_CROPPED==============
--=====================================================================APPLY_NET============
function applyNet(model,iimage,detections,isizeIn,isizeOut,threshold)
   if(detections ~= nil) then
   local iimVec = torch.Tensor(detections:size(1),ich,isizeIn,isizeIn)
   local oimVec = torch.Tensor(detections:size(1),ich,isizeOut,isizeOut)
   for i=1,detections:size(1) do
		detect=detections[i]
		iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iimage, isizeIn)
		oimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iimage, isizeOut)
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
   iimVec1:index(oimVec,1,k)
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
--=====================================================================END_APPLY_NET=========
--========================================NMS
function nms(boxes, overlap)--nms(boxes, overlap,s)
  
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
  area:map2(x2,x1,function(xx,xx2,xx1) return xx2-xx1+1 end)  -- area=Dx
  area:map2(y2,y1,function(xx,xx2,xx1) return xx*(xx2-xx1+1) end) -- area=area*Dy

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
    xx1:clamp(x1[i],x2[i])
    yy1:index(y1,1,I)
    yy1:clamp(y1[i],y2[i])
    xx2:index(x2,1,I)
    xx2:clamp(x1[i],x2[i])
    yy2:index(y2,1,I)
    yy2:clamp(y1[i],y2[i])
    
    w:resizeAs(xx2):zero()
    w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
    h:resizeAs(yy2):zero()
    h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
    
    local inter = w
    inter:cmul(h)

    local o = h
    local a = h
    xx1:index(area,1,I)
    torch.cdiv(o,inter,xx1+area[i]-inter)
    --a=xx1*area[i]
    --a=a:apply(function(aa) return math.sqrt(aa) end)
    --m=xx1+area[i]
    --m=m/2
    --a=a:map(m,function(aa,mm) return aa/mm end)
    I = I[o:le(overlap)]--I[o:le(overlap) or a:le(s)]
    --I = I[o:le(overlap) or a:ne(1)]--(keepifI==1);I= I[o:le(overlap) or a:gt(1 + si) or a:lt(a - si)]

  end

  pick = pick[{{1,counter-1}}]
  return pick
end
--================================================END_NMS
--============================================GLOBAL_NMS===========================
function gnms(boxes, overlap)
  
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
    xx1:clamp(x1[i],x2[i])
    yy1:index(y1,1,I)
    yy1:clamp(y1[i],y2[i])
    xx2:index(x2,1,I)
    xx2:clamp(x1[i],x2[i])
    yy2:index(y2,1,I)
    yy2:clamp(y1[i],y2[i])
    
    w:resizeAs(xx2):zero()
    w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
    h:resizeAs(yy2):zero()
    h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
    
    local inter = w
    inter:cmul(h)

    local o = h
    xx1:index(area,1,I)
    torch.cdiv(o,inter,xx1+area[i]-inter)
    I = I[o:le(overlap)]
  end

  pick = pick[{{1,counter-1}}]
  return pick
end
--===============================================================END_GLOBAL_NMS===============


network_fov = 12--12
network_sub = 2 --2
step_width = network_sub
step_height = network_sub
----------------------------------PROCESSP---------------------------------------------------------------------------------------
function processp(imAdd,scaleidx)
   -- grab frame
   --frame = camera:forward()
   --frame = image.load('18.jpg')
local windsperscale=0
   print(imAdd..'/scaleidx='..scaleidx)
   local frame = image.load(inFolder..imAdd)
   print(frame:size())
--if (frame:size(2)<100 and frame:size(3)<100 )  then 
--return 
--collectgarbage()
--end
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

  if (nScales==nil)  then
collectgarbage()
return 
  end

   det = net12:forward(pyramid)


   distributions = unpacker:forward(det, coordinates)

   local rawresults = {}

   -- function FFI:
   for i,distribution in ipairs(distributions) do
       parseFFI(distribution, threshold12, rawresults, nScales[i])
   end

   -- (7) clean up results
  if(#rawresults <1) then
	collectgarbage()
	  return 
   end
   local detections1 = torch.Tensor(#rawresults,5)
   for i,res in ipairs(rawresults) do

      local scale = res[3]
	--print('scale='..scale)
      local x = res[1]*network_sub/scale
	--print('x='..x)
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

   
	print("net12 done")

   --================Calibration 12 Net===================
   detections1 = applyCalibNet(calib12,imGray,detections1,FilterSize1,threshold12Calib)
   print("net12-calib done")
if(useNMS==1) then
		keep1nms = nms(detections1,nmsth1)--nms(detections1,nmsth1,si)
	else
		keep1nms = torch.LongTensor(detections1:size(1))
		for j=1,keep1nms:size(1) do
			keep1nms[j] = j
		end
   end
print("nms done")
   local ndetections1 = torch.Tensor()
   ndetections1:index(detections1,1,keep1nms)
   detections1 = ndetections1--detections1 = ndetections1
   ndetections1 = nil


--[[
   --======================24 Net=========================
   detectionsT,out2,imVec = applyNet(net24,imGray,detections1,FilterSize2,FilterSize2,threshold24)
   print("net24 done")
   --================Calibration 24 Net===================
   --detections2 = applyCalibNetCropped(calib24,imVec,detectionsT,out2,FilterSize2,threshold48Calib,useNMS,nms2th)
   detections1 = applyCalibNetCropped(calib24,imVec,detectionsT,out2,FilterSize2,threshold24Calib,useNMS,nmsth2,0)--,si)--
   print("net24-calib done")
   --print("net24 done")   
--]]
--======================= SAVE FILE  ===================
if detections1~=nil then
keep1nms = torch.LongTensor(detections1:size(1))
		for j=1,keep1nms:size(1) do
			keep1nms[j] = j
		end
       for icnt=1,keep1nms:size(1) do
		detect=detections1[icnt]--detections1[keep1nms[i]]
		im2,bbfg = cropImagebb({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, frame, FilterSize3)--
                --im2 = cropImagenonorm({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, frame, FilterSize3)--
		if bbfg==0 then
	
		im2=image.rgb2y(im2)
		windsperscale=windsperscale+1   --TODO if bbfg is used then skip count if black box occurs in detection window
		image.save('AFLW_FACELESS_PATCHES/'..string.sub(imAdd,1,string.find(imAdd,'.jpg')-1)..'_'..windsperscale..'_'..scaleidx..'.jpg',im2)
		end

	end

end

--Offs=i+Offs



collectgarbage()
return 
end
 ----------------------------------------------------------------------------------END_PROCESSP---------------------------------------

-- setup camera
--camera = image.Camera(0)

--scales = {1}
-- use a pyramid packer/unpacker
require 'PyramidPacker'
require 'PyramidUnPacker'



address = loadDataFiles(inFolder)
for l=1,Nscales do                            
packer = nn.PyramidPacker(net12, {scales[l]})
unpacker = nn.PyramidUnPacker(net12)
	for i=1,#address do
		processp(address[i],l)
	end
end
