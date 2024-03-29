require 'torch'
require 'nn'
require 'nnx'
require 'camera'
require 'cunn'
require 'optim'
require 'math'
require 'sys'
-- Local files
require 'demo/PyramidPacker'
require 'demo/PyramidUnPacker'

torch.manualSeed(1234)

ich = 3
TestSetno=2;
Nth=15
max_ImSize=700
min_size = 35
--Validation set index 1:2000



  stat12net={}
  stat12calib={}

  gray_path='/home/jblan016/FaceDetection/Cascade/GrayCascadeNet'
  rgb_path='/home/jblan016/FaceDetection/Cascade/RGBCascadeNet'
if ich == 1 then
  net12_dir = gray_path..'/20net/results/'
  calib12_dir =gray_path..'/12calibnet/results/'
  net48_dir = gray_path..'/48net/results/'
  calib48_dir =gray_path..'/48calibnet/results/'
elseif ich ==3 then
  net12_dir = rgb_path..'/20net/results/'
  calib12_dir =rgb_path..'/12calibnet/results/'
  net48_dir = rgb_path..'/48net/results/'
  calib48_dir =rgb_path..'/48calibnet/results/'
end

stat12net.mean={}
stat12net.std={}
stat48net={}
stat48net.mean={}
stat48net.std={}
if ich ==3 then
  for i=1,ich do
    stat12net.mean[i]=torch.load(net12_dir..'mean.dat')[i]
    stat12net.std[i]=torch.load(net12_dir..'std.dat')[i]
    stat48net.mean[i]=torch.load(net48_dir..'mean.dat')[i]
    stat48net.std[i]=torch.load(net48_dir..'std.dat')[i]
  end
elseif ich==1 then
  stat12net.mean=torch.load(net12_dir..'mean.dat')[1]
  stat12net.std=torch.load(net12_dir..'std.dat')[1]
  stat48net.mean=torch.load(net48_dir..'mean.dat')[1]
  stat48net.std=torch.load(net48_dir..'std.dat')[1]
end
stat12calib.mean={}
stat12calib.std={}
stat48calib={}
stat48calib.mean={}
stat48calib.std={}
if ich ==3 then
  for i=1,ich do
    stat12calib.mean[i]=torch.load(calib12_dir..'mean.dat')[i]
    stat12calib.std[i]=torch.load(calib12_dir..'std.dat')[i]
    stat48calib.mean[i]=torch.load(calib48_dir..'mean.dat')[i]
    stat48calib.std[i]=torch.load(calib48_dir..'std.dat')[i]
  end
elseif ich==1 then
  stat12calib.mean=torch.load(calib12_dir..'mean.dat')[1]
  stat12calib.std=torch.load(calib12_dir..'std.dat')[1]
  stat48calib.mean=torch.load(calib48_dir..'mean.dat')[1]
  stat48calib.std=torch.load(calib48_dir..'std.dat')[1]
end

  net12_path = net12_dir..'model.net'
  calib12_path = calib12_dir..'model1.net'
  net48_path = net48_dir..'model.net'
  calib48_path = calib48_dir..'model1.net'

  database_path='/home/jblan016/FaceDetection/Cascade/dataset/data/0/'
network_fov = {20,20}
network_sub = 4

FilterSize1 = {20,20}
FilterSize12cnet = {12,12}
FilterSize2 = {48,48}
FilterSize48cnet = {48,48}

threshold12 = -1    -- To keep all detections
if ich==1 then
  classth1=0.15
elseif ich==3 then
  classth1=0.4
end
threshold12Calib = .45

nmsth1 = .95
nmsth2 = .95

useNMS = 0 --un 


scale_min = network_fov[1]/min_size
iou_th = 0.5
nscales = 1000
scratio = 1/math.sqrt(math.sqrt(2))

function pyramidscales(n_max,FilterSize,MinFaceSize,sc,imW,imH)  
  --Outputs the number of pyramid levels and the scaling factors given (FilterSize,Minimum Face Size,scaling factor step,image Height,image Width)   
  --scaling factor  0<sc<1  

  local s=FilterSize/MinFaceSize --initial default scale
  local Npl=math.floor(-math.log(math.min(imW,imH)/MinFaceSize)/math.log(sc))+1 --Max Number of Pyramid Levels(Npl) given image width height, initial scaling "s" and scaling level ratio "sc" (smaller than 1)
  Npl=math.min(n_max,Npl)
  local scales={}

--Fs=torch.Tensor(Npl)
  for k=1,Npl do
    scales[k]=s*(sc)^(Npl-k)
  --  Fs[k]=MinFaceSize/(sc)^(Npl-k)-- Apparent filtersize
  end
  return scales, Npl
end



--== DEBUG FUNCTION

function print_r ( t )  
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        print(indent.."["..pos.."] => "..tostring(t).." {")
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        print(indent..string.rep(" ",string.len(pos)+6).."}")
                    elseif (type(val)=="string") then
                        print(indent.."["..pos..'] => "'..val..'"')
                    else
                        print(indent.."["..pos.."] => "..tostring(val))
                    end
                end
            else
                print(indent..tostring(t))
            end
        end
    end
    if (type(t)=="table") then
        print(tostring(t).." {")
        sub_print_r(t,"  ")
        print("}")
    else
        sub_print_r(t,"  ")
    end
    print()
end

function pause()
print('press enter to continue')
   io.stdin:read'*l'
end
---- END DEBUG FUNCTIONS

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

function IoU(d,D) --compare ioU of d with D. d is a row vector and D are a row ordered list of bounding box coordinates
  local N=D:size(1)
  local InoU=torch.Tensor(N):fill(0) --intersection for now
  local O=torch.Tensor(N):fill(0) --overlap
  local dswap=torch.Tensor(N):fill(0)
  local xmin2=torch.cmin(D[{{},3}],dswap:fill(d[3]))
  local xmax1=torch.cmax(D[{{},1}],dswap:fill(d[1]))
  local dx=torch.cmax(xmin2-xmax1,dswap:fill(0))
  local Xfit=dx:ge(0)
  local ymin2=torch.cmin(D[{{},4}],dswap:fill(d[4]))
  local ymax1=torch.cmax(D[{{},2}],dswap:fill(d[2]))
  local dy=torch.cmax(ymin2-ymax1,dswap:fill(0))
  local Yfit=dy:ge(0)
  local XYfit=torch.cmul(Xfit,Yfit)--Xfit:eq(Yfit)  --XYfit =(Xfit AND Yfit)
  --compute intersectionarea
  InoU[XYfit]=torch.cmul(dx[XYfit]+1,dy[XYfit]+1) 

  --compute overlap
  O[XYfit]=torch.cmul(D[{{},3}][XYfit]-D[{{},1}][XYfit]+1,D[{{},4}][XYfit]-D[{{},2}][XYfit]+1)+math.pow(d[3]-d[1]+1,2)-InoU[XYfit] --compute only overlaps for non-zero intersections
  InoU[XYfit]=torch.cdiv(InoU[XYfit],O[XYfit])
  return InoU, XYfit
end

function frame2im(fr,ich)
  if ich==1 then
    if fr:size(1)==1 then
    img = fr:clone()
    elseif fr:size(1)==4 then
    img = fr[{{1,3},{},{}}]:clone()
    img=image.rgb2y(img)
    elseif fr:size(1)==3 then
    img=image.rgb2y(fr):clone()
    end
  elseif ich==3 then
    if fr:size(1)==4 then
      img = fr[{{1,3},{},{}}]:clone()
    elseif fr:size(1) == 1 then 
      img=torch.repeatTensor(fr,ich,1,1)
    elseif fr:size(1)==3 then
      img=fr:clone()
    end
  end
return img
end

function normalize(im,stats,ich)
   if ich==3 then
    for i = 1, ich do
        im[{i, {}, {}}]:add(-stats.mean[i])
        im[{i, {}, {}}]:div(stats.std[i])
    end
    elseif ich==1 then
      im:add(-stats.mean)
      im:div(stats.std)
    end
    return im
end


function cropImage(iarg, iim, isize)
    -- This function crops an image(CxHxW) with input arguments (x1,y1,x2,y2) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
    -- The processing is done in matrix coordinates (i,j).
    local img_width = iim:size(3)
    local img_hight = iim:size(2)

    local oim = torch.Tensor(ich, iarg.y2-iarg.y1+1, iarg.x2-iarg.x1+1):fill(0)
    local Mask = torch.ByteTensor(ich, iarg.y2-iarg.y1+1, iarg.x2-iarg.x1+1):fill(0)
    local iIx2 = math.min(iarg.x2, img_width)
    local iIx1 = math.max(iarg.x1, 1)  -- 1 because start index of image=1
    local DiIx = iIx2 - iIx1 + 1  -- span of elements in x copyied from input image to output image
    local Offsx = iIx1 - math.min(iarg.x1, iIx1) -- index offset of output image in x

    local iIy2 = math.min(iarg.y2, img_hight)
    local iIy1 = math.max(iarg.y1, 1)  -- 1 because start index of image=1
    local DiIy = iIy2 - iIy1 + 1  -- span of elements in y copyied from input image to output image
    local Offsy = iIy1 - math.min(iarg.y1, iIy1) -- index offset of output image in y

    local Ic = iim:sub(1, -1, iIy1, iIy2, iIx1, iIx2):clone()
    Mask:sub(1, -1, Offsy+1, Offsy+DiIy, Offsx+1, Offsx+DiIx):fill(1)
    oim:maskedCopy(Mask, Ic)
    oim = image.scale(oim, isize[1], isize[2],'bicubic')
    return oim          
end



function applyNet(model, iimage, detections, isize, threshold,ich)
    if(detections ~= nil) then
        local iimVec = torch.Tensor(detections:size(1), ich, isize[2], isize[1])
        for i = 1, detections:size(1) do
            local detect = detections[i]
            iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iimage, isize)
        end
      
        o = torch.exp(model:forward(iimVec:cuda())):double()
        local cnt = 1
        k = torch.LongTensor(o:size(1))
        for i = 1, o:size(1) do
            a = o[i][1]
            --b = out2[i][2]
            if (a > threshold) then
                k[cnt] = i
                cnt = cnt + 1
            end 
        end
        if(cnt > 1) then
            k = k[{{1, cnt-1}}]

            local o1 = torch.Tensor()
            o1:index(o, 1, k)

            local iimVec1 = torch.Tensor()
            iimVec1:index(iimVec, 1, k)

            local detectionsR = torch.Tensor()
            detectionsR:index(detections, 1, k)

            detectionsR[{{},5}] = o1[{{},1}]    -- Set new confidence values

            return detectionsR, o1, iimVec1
        else
            return nil, nil, nil
        end
    else
        return nil, nil, nil
    end
end



function applyCalibNet(model,iim,detections,isize,threshold,ich)
    if(detections~=nil) then
        local iimVec = torch.Tensor(detections:size(1),ich,isize[2],isize[1])
        for i=1,detections:size(1) do
            detect=detections[i]
            iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iim, isize)

        end
        --image.display{image=im2, nrow=32}
        local ca1 = torch.exp(model:forward(iimVec:cuda())):float()  -- ca1_(#det x 45 )
        --local ca1 = torch.exp(model:forward(iimVec:float()))  -- ca1_(#det x 45 )
        local trans = torch.Tensor(ca1:size(1),3):zero()
        local c=0
        for i=1,ca1:size(1) do-- for each detection window 
            c = 0
            local maxlbl=torch.max(ca1[i])
            for j=1,ca1:size(2) do -- for all 45 labels
                if(ca1[i][j] > threshold) then
                    trans[i] = trans[i] + T[j]
                    c = c + 1
                end
            end
            if (c~=0) then --TODO pick a better way of choosing max
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

        local w = 0 
        local h = 0
        local sn = 0
        local xn = 0
        local yn = 0
        local x1 = 0
        local y1 = 0
        local x2 = 0
        local y2 = 0
        for i=1,trans:size(1) do 
            w = math.abs(detections[i][3]-detections[i][1])+1--w = detections[i][3]-detections[i][1]  
            h = math.abs(detections[i][4]-detections[i][2])+1--h = detections[i][4]-detections[i][2]
            sn = trans[i][1]
            xn = trans[i][2]
            yn = trans[i][3]
            x1 = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn)
            y1 = torch.round(detections[i][2] -w*yn+(sn-1)*h/2/sn)
            x2 = torch.round(detections[i][1] -w*xn+(sn-1)*w/2/sn+w/sn)
            y2 = torch.round(detections[i][2] -w*yn+(sn-1)*h/2/sn+h/sn)
            detections[i][1] = x1
            detections[i][2] = y1
            detections[i][3] = x2
            detections[i][4] = y2
        end
    end
    return detections
end

function concatdets(detsMSc)
  local Nsc = #detsMSc
  local scale_lim = torch.Tensor(Nsc+1):fill(0)
  local Ntot = 0

  for i=1,Nsc do 

    Ntot=detsMSc[i]:size(1)+Ntot
    
    scale_lim[i+1] = detsMSc[i]:size(1)+scale_lim[i]
  end
  local dets = torch.Tensor(Ntot,5)

  for i=1,Nsc do dets[{{1+scale_lim[i],scale_lim[i+1]},{}}] = detsMSc[i] end
  return dets
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



--TODO changer ceci
testAddBB=torch.load('./Sets/Set'..TestSetno..'.dat') --in format print(testAddBB['address'][1])print(testAddBB['address' OR 'bb'][k])


pos_num=#testAddBB['address']
testImgs=testAddBB['address']
pos_num=10

network_temp = torch.load(net12_path)
print(network_temp)
net12 = nn.Sequential()

for i=1,8 do
    net12:add(network_temp.modules[i])
end
classifier1 = nn.Sequential()
for i=9,11 do 
    classifier1:add(network_temp.modules[i])
end


classifier = nn.SpatialClassifier(classifier1)
net12:add(classifier)
net12:cuda()

calib12 = torch.load(calib12_path)
calib12:cuda()

calib48 = torch.load(calib48_path)
calib48:evaluate()
calib48:cuda()
net48 = torch.load(net48_path)
net48:evaluate()
net48:cuda()
-- use a pyramid packer/unpacker --moved to 330
--packer = nn.PyramidPacker(net12, scales)  -- Possiblement forcer ceci ailleurs vu que scales est dynamique et packer est cst
--unpacker = nn.PyramidUnPacker(net12)
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

cth=90
threshold48Calib = cth/100
  local testLogger = optim.Logger('./logs48/2048net_'..ich..'channels_48ct_calib2Th'..cth..'.log')
  detPred = {}
  detWin = {}
local IM12C={}
local IM48={}
local IM48C={}

  for j=1,pos_num do -- in "images in dataset" do
    -- grab frame
    --print(database_path..testImgs[j],j)
    frame = image.load(database_path..testImgs[j])

    
    if math.max(frame:size(2),frame:size(3))>max_ImSize then
      local resz = max_ImSize/math.max(frame:size(2),frame:size(3))
      frame=image.scale(frame,torch.round(resz*frame:size(3)),torch.round(resz*frame:size(2)),'bicubic')
      testAddBB['bb'][j][1][1]=torch.round((testAddBB['bb'][j][1][1]-1)*resz+1)
      testAddBB['bb'][j][1][2]=torch.round((testAddBB['bb'][j][1][2]-1)*resz+1)
      testAddBB['bb'][j][1][3]=torch.round(testAddBB['bb'][j][1][3]*resz)
      testAddBB['bb'][j][1][4]=torch.round(testAddBB['bb'][j][1][4]*resz)
      --print(frame:size(1),frame:size(2),frame:size(3))
    end

    local scales = {}
    local EffNScales = 0
    -- use a pyramid packer/unpacker
    scales,EffNScales = pyramidscales(nscales,FilterSize1[1],min_size,scratio,frame:size(3),frame:size(2))
    print('j='..j,'#Scales='..EffNScales)
    packer = nn.PyramidPacker(net12, scales)  
    unpacker = nn.PyramidUnPacker(net12)
    

    -- process input at multiple scales
    local im=frame2im(frame,ich)

    -- Normalise mean and variance for each channel
    im12=im:clone()
    im12=normalize(im12,stat12net,ich)
    pyramid, coordinates = packer:forward(im12)    -- Why does this have to be a double when it is not in the other script
    im12=nil
    --print('--------------------------')
    --print(pyramid:size())
    tt = torch.Timer() 
    det = net12:forward(pyramid:cuda())
    pyramid=nil
    tt:stop()
    distributions = unpacker:forward(det, coordinates)
    rawresults = {}
    -- function FFI:
    
    for i,distribution in ipairs(distributions) do
        parseFFI(distribution, threshold12, rawresults, scales[i])
    end
    if(#rawresults < 1) then
        error('No detections')
    end
    -- (7) clean up results
    detections1 = torch.Tensor(#rawresults,5)
    
    --print(#rawresults)
    for i,res in ipairs(rawresults) do
        local scale = res[3]
        local x = torch.round(res[1]*network_sub/scale)
        local y = torch.round(res[2]*network_sub/scale)
        local w = torch.round(network_fov[1]/scale)
        local h = torch.round(network_fov[2]/scale)
        detections1[{i,1}] = x
        detections1[{i,2}] = y
        detections1[{i,3}] = x + w 
        detections1[{i,4}] = y + h
        detections1[{i,5}] = res[4]
    end
    --print("net12 done")

    rawresults=nil
    local detperscale = torch.Tensor(EffNScales)
    local scale_lim = torch.Tensor(EffNScales+1):fill(0)
    for i=1,EffNScales do detperscale[i] = distributions[i]:size(2)*distributions[i]:size(3) end
    detmsc={}
    if EffNScales>1 then
      for i=1,EffNScales do scale_lim[i+1] = detperscale[i]+scale_lim[i] end
      for i=1,EffNScales do 
      detmsc[i]=detections1[{{1+scale_lim[i],scale_lim[i+1]},{}}]:clone() --ici n'est jamais nul
      end
    end
     detWin[j]={}
     detWin[j]=detmsc
     IM12C[j]=im:clone()
     IM12C[j]=normalize(IM12C[j],stat12calib,ich)
     IM48C[j]=im:clone()
     IM48C[j]=normalize(IM48C[j],stat48calib,ich)
     IM48[j]=im:clone()
     IM48[j]=normalize(IM48[j],stat48net,ich)
  end -- end imagelist loop
  detmsc=nil
  
  --SECOND PART
  local tempdets=torch.Tensor()
  local mask=torch.Tensor() 
  local frame=torch.Tensor()
  local EffNScales=0
  local fp = 0
  local tp = 0
  local fn = 0
  local gt = {}
  local windows = 0
  local detections1=torch.Tensor()
  local iou=torch.Tensor()
  local ispos=torch.Tensor()
  local isneg=torch.Tensor()
  local labeledpos=torch.Tensor()
  local labeledneg=torch.Tensor()
  local detsmc={}
  

--local th= torch.cat(torch.range(100,98,-.05),torch.range(98.25,95,-.25))
--      th= torch.cat(th,torch.range(96,0,-1))
local th= torch.cat(torch.range(55,10,-5),torch.range(11,1,-1))
local Nth=(#th)[1]
for nth=1,Nth do
  classth=th[nth]/100-.0001
  print(th[nth])
  fp = 0
  tp = 0
  fn = 0
  gt = {}
  windows = 0
  detections1=torch.Tensor()
  iou=torch.Tensor()
  ispos=torch.Tensor()
  isneg=torch.Tensor()
  labeledpos=torch.Tensor()
  labeledneg=torch.Tensor()

  for j=1,pos_num do


   
    tempdets=torch.Tensor()
    mask=torch.Tensor() 

    EffNScales=(#detWin[j])
      detsmc={}

    --print(frame:size(1),frame:size(2),frame:size(3))
   


     
 ---------------------------Calib12
      if cth>0 then
        for i=1,EffNScales do
          mask=detWin[j][i][{{},5}]:ge(classth1)
          detsmc[i]=detWin[j][i]:clone()
          if torch.sum(mask)==0 then
          --print('skipped index '..i)
          goto skip12c
          end
          tempdets=detWin[j][i][torch.repeatTensor(mask,5,1):t()]:clone()
        
          tempdets=tempdets:resize((#tempdets)[1]/5,5)
          tempdets=applyCalibNet(calib12,IM12C[j],tempdets,FilterSize12cnet,threshold12Calib,ich) 
         
          detsmc[i][torch.repeatTensor(mask,5,1):t()]=tempdets:resize((#tempdets)[1]*5)
          collectgarbage()
          tempdets=nil
          mask=nil
          ::skip12c::
        
        end
      else
        for i=1,EffNScales do
          detsmc[i]=detWin[j][i]:clone()
        end
      end

      --print('12calib done')
----------------------nms
      local tempdetsPOS=torch.Tensor()
      local tempdetsNEG=torch.Tensor()
      if nmsth1>0 then
        for i=1,EffNScales do
          mask=detWin[j][i][{{},5}]:ge(classth1)
          mask2=detWin[j][i][{{},5}]:lt(classth1)
          detsmc[i]=detWin[j][i]:clone()
          sm=torch.sum(mask)
          sm2=torch.sum(mask2)
          if sm>0 then
          tempdetsPOS=detWin[j][i][torch.repeatTensor(mask,5,1):t()]:clone()
          tempdetsPOS=tempdetsPOS:resize((#tempdetsPOS)[1]/5,5)
          tempdetsPOS,crap=nms(tempdetsPOS,nmsth1)
          crap=nil
          else 
          goto skipnms1
          end
          if sm2>0 then
          tempdetsNEG=detWin[j][i][torch.repeatTensor(mask2,5,1):t()]:clone()
          tempdetsNEG=tempdetsNEG:resize((#tempdetsNEG)[1]/5,5)
          end
          if sm2>0 and sm>0 then
          tempdets=torch.cat(tempdetsPOS,tempdetsNEG,1)
          elseif sm>0 and sm2==0 then 
          tempdets=tempdetsPOS
          elseif sm2>0 and sm==0 then
          tempdets=tempdetsNEG
          end
          detsmc[i][torch.repeatTensor(mask,5,1):t()]=tempdets:resize((#tempdets)[1]*5)
          collectgarbage()
          ::skipnms1::
          tempdets=nil
          tempdetsPOS=nil
          tempdetsNEG=nil
          mask=nil
          mask2=nil
          collectgarbage()
        end
      end
  --print('nms done')
      detections1=concatdets(detsmc)
----------------------second classif
          firstscores=detections1[{{},5}]:clone()
          mask=detections1[{{},5}]:ge(classth1)
          if torch.sum(mask)==0 then
          --print('skipped index '..i)
          tempdets=nil
          mask=nil
          goto skipn2
          end
          tempdets=detections1[torch.repeatTensor(mask,5,1):t()]:clone()
          tempdets=tempdets:resize((#tempdets)[1]/5,5)

          tempdets=applyNet(net48,IM48[j],tempdets,FilterSize2,-1,ich)
          detections1[torch.repeatTensor(mask,5,1):t()]=tempdets:resize((#tempdets)[1]*5)
          collectgarbage()
          tempdets=nil
          mask=nil
          ::skipn2::

---------------------------Calib48
      if threshold48Calib>0 then
          mask=(detections1[{{},5}]:ge(classth)+firstscores:ge(classth1)):eq(2)
          if torch.sum(mask)==0 then
          --print('skipped index '..i)
          tempdets=nil
          mask=nil
          goto skipcalib48
          end
          tempdets=detections1[torch.repeatTensor(mask,5,1):t()]:clone()
          tempdets=tempdets:resize((#tempdets)[1]/5,5)
          tempdets=applyCalibNet(calib48,IM48C[j],tempdets,FilterSize48cnet,threshold48Calib,ich) 
          detections1[torch.repeatTensor(mask,5,1):t()]=tempdets:resize((#tempdets)[1]*5)
          collectgarbage()
          tempdets=nil
          mask=nil
          ::skipcalib48::
       end

 --------------------------------


      --pause()
            gt[j] = torch.Tensor{testAddBB['bb'][j][1][1],testAddBB['bb'][j][1][2],testAddBB['bb'][j][1][1]+testAddBB['bb'][j][1][3],testAddBB['bb'][j][1][2]+testAddBB['bb'][j][1][4]}
            iou=IoU(gt[j],detections1[{{},{1,4}}])
            --m,ii=torch.max(iou,1)
            
            ispos=iou:ge(iou_th)
            isneg=iou:lt(iou_th)
            
            labeledpos=(detections1[{{},5}]:ge(classth)+firstscores:ge(classth1)):eq(2)
            labeledneg=(detections1[{{},5}]:lt(classth)+firstscores:ge(classth1)):ge(1)
            
            tp=tp+torch.sum(torch.add(ispos,labeledpos):eq(2))
            
            fp=fp+torch.sum(torch.add(isneg,labeledpos):eq(2))
            
            fn=fn+torch.sum(torch.add(ispos,labeledneg):eq(2))
            windows=windows+detections1:size(1)
            --print(iou)
            --print(detWin[j][{{},5}])
            print('tp='..tp,'fp='..fp,'fn='..fn,'recall='..tp/(tp+fn),'precision='..tp/(tp+fp),'th='..classth,'j='..j)
            --pause()
            detections1=nil
            firstscores=nil
            collectgarbage()
            
  end -- end imagelist loop


    testLogger:add{['% Threshold'] = classth,
                   ['% True positive'] = tp,
                   ['% False positive'] = fp,
                   ['% False negative'] = fn,
                   ['% Window number']=windows,
                   ['% Precision']=tp/(tp+fp),
                   ['% Recall']=tp/(tp+fn),
                   ['% F Score']=1/(1+2*(tp/(fp+fn)))} --survived windows because nms removes some
                   collectgarbage()
end --end threshold loop



