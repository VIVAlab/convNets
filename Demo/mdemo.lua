--require 'xlua'
require 'torch'
require 'nn'
require 'nnx'
--require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'cunn'
--require 'inn'

-- Local files


camuse=1
if camuse==1 then
   camera = image.Camera(0)
   GUI = 'g.ui'
end
norm_data={}
ich = 1
   stat12net={}
  stat12calib={}
  gray_path='/home/jblan016/FaceDetection/Cascade/GrayCascadeNet'
  rgb_path='/home/jblan016/FaceDetection/Cascade/RGBCascadeNet'
if ich == 1 then
  filenameFN='12net'--'20netNoFC'--'20net2 (copy)'
  net12_dir = gray_path..'/'..filenameFN..'/results/'
  calib12_dir =gray_path..'/20calibnet/results/'
  net48_dir = gray_path..'/48net2/results/'
  calib48_dir =gray_path..'/48calibnet/results/'
elseif ich ==3 then
  net12_dir = rgb_path..'/20net/results/'
  calib12_dir =rgb_path..'/12calibnet/results/'
  net48_dir = rgb_path..'/48net/results/GoodResults/'
  calib48_dir =rgb_path..'/48calibnet/results/'
end
if string.match(filenameFN, "20") then
    require 'PyramidPacker'
    require 'PyramidUnPacker'
    network_fov = {20,20}
    network_sub = 4
else
    require 'PyramidPacker12net'
    require 'PyramidUnPacker12net'
    network_fov = {12,12}
    network_sub = 2
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
if string.match(filenameFN, "12") then
    net12_path = net12_dir..'modelNoBN.net'
else
  net12_path = net12_dir..'model.net'
end
  
  calib12_path = calib12_dir..'model1.net'
  net48_path = net48_dir..'model.net'
  calib48_path = calib48_dir..'model1.net'
--net12_path='/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/12net/results/modelNoBN.net'
imgcnt = 1




--FilterSize16 = {12,12}
FilterSize1 = {20,20}
FilterSize12 = {12,12}
FilterSize48 = {48,48}
nmsth1 = .9
nmsth2 = .2
gnmsth = .2

useNMS = 1 --un 
useGNMS = 1

min_size = 35--180
scale_min = network_fov[2]/min_size

nscales = 10
scratio = 1/math.sqrt(2)
--[[
--[[
scales={}
for k=1,nscales do
    scales[k]=scale_min*(scratio)^(nscales-k)
  print(scales[k])
  end
--]]
function KeepPositives(detmsc,threshold12)
  local mask = torch.Tensor()
  local Ndets=#detmsc
  local newdetmsc={}
  local r=0
  local masksum=0
  for i=1,Ndets do
    mask=detmsc[i][{{},5}]:ge(threshold12)
    masksum=torch.sum(mask)
    if masksum>0 then
    r=r+1
    newdetmsc[r]=detmsc[i][torch.repeatTensor(mask,5,1):t()]:clone()
    newdetmsc[r]=newdetmsc[r]:resize((#newdetmsc[r])[1]/5,5)
    end
  end
  return newdetmsc,r
end

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


-- display function
function display()
    zoom = 1
    win:gbegin()
    win:showpage()
    image.display{image=frame, win=win, zoom=zoom}
--    image.display{image=testim, win=win, zoom=zoom}
    
    if widget.s1.checked then
        s1 = 1
    else
        s1 = 0
    end
    if widget.s2.checked then
        s2 = 1
    else
        s2 = 0
    end
    if widget.s3.checked then
        s3 = 1
    else
        s3 = 0
    end
    if widget.s4.checked then
        s4 = 1
    else
        s4 = 0
    end
    if widget.s5.checked then
        s5 = 1
    else
        s5 = 0
    end
    if widget.s6.checked then
        s6 = 1
    else
        s6 = 0
    end
    if s1==1 then
        if(detections1~=nil) then
            win:setcolor(0,0,1)
            win:setlinewidth(1)  
            for i=1,detections1:size(1) do
                detect=detections1[i]:narrow(1,1,5)
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s2==1 then
        if(detections2~=nil) then
            win:setcolor(1,0,0)
            win:setlinewidth(1)
            for i=1,detections2:size(1) do
                detect=detections2[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s3==1 then
        if(detections3~=nil) then
            win:setcolor(0,1,0)
            win:setlinewidth(2)
            for i=1,detections3:size(1) do
                detect=detections3[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s4==1 then
        if(detections4~=nil) then
            win:setcolor(1,0,0)
            win:setlinewidth(2)
            for i=1,detections4:size(1) do
                detect=detections4[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s5==1 then
        if(detections5~=nil) then
            win:setcolor(0,1,1)
            win:setlinewidth(2)
            for i=1,detections5:size(1) do
                detect=detections5[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s6==1 then
        if(detections6~=nil) then
            win:setcolor(0,1,0)
            win:setlinewidth(2)
            for i=1,detections6:size(1) do
                detect=detections6[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
--    for i=1,#t do
--        win:setcolor(1,0,0)
--        win:rectangle(t[i][2]*network_sub, t[i][1]*network_sub, 64, 64)
--        win:stroke()
--        win:setfont(qt.QFont{serif=false,italic=false,size=16})
--        win:moveto(detect.x, detect.y-1)
--        win:show('face')
--    end
    win:gend()
    detections1 = nil
    detections2 = nil
    detections3 = nil   
    detections4 = nil
    detections5 = nil
    detections6 = nil  
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


function cropImage(iarg, iim, isize)
    -- This function crops an image(CxHxW) with input arguments (x1,y1,x2,y2) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
    -- The processing is done in matrix coordinates (i,j).
    local img_width = iim:size(3)
    local img_hight = iim:size(2)

    local oim = torch.Tensor(ich, iarg.y2 - iarg.y1 + 1, iarg.x2 - iarg.x1 + 1):fill(0)
    local Mask = torch.ByteTensor(ich, iarg.y2 - iarg.y1 + 1, iarg.x2 - iarg.x1 + 1):fill(0)
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

    oim = image.scale(oim, isize[1], isize[2])
    
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

function applyCalibNet(model,iim,detections,isize,threshold,ich)
    if(detections~=nil) then
        local iimVec = torch.Tensor(detections:size(1),ich,isize[2],isize[1])
        for i=1,detections:size(1) do
            detect=detections[i]
            iimVec[i] = cropImage({x1 = detect[1], y1 = detect[2], x2 = detect[3], y2 = detect[4]}, iim, isize)
        end
        --image.display{image=im2, nrow=32}
        local ca1 = torch.exp(model:forward(iimVec:cuda())):double()  -- ca1_(#det x 45 )
        local trans = torch.Tensor(ca1:size(1),3):zero()
    

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
   
   
        for i=1,trans:size(1) do 
            w = math.abs(detections[i][3]-detections[i][1])--w = detections[i][3]-detections[i][1]  
            h = math.abs(detections[i][4]-detections[i][2])--h = detections[i][4]-detections[i][2]
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


-- process function
function process()
    -- grab frame
    if camuse==1 then
   frame = camera:forward()
    else
    frame = image.load(t[imgcnt])
    end
    print(t[imgcnt])
    -- process input at multiple scales    
    im = frame2im(frame,ich)
    scales,EffNScales=pyramidscales(nscales,network_fov[2],min_size,scratio,frame:size(3),frame:size(2))
    if EffNScales<1 then
    goto skip2end
    end
    packer = nn.PyramidPacker(net12, scales)  
    unpacker = nn.PyramidUnPacker(net12)

    -- Normalise mean and variance for each channel
    im12=im:clone()
    im12=normalize(im,stat12net,ich)

    pyramid, coordinates = packer:forward(im12)    -- Why does this have to be a double when it is not in the other script
    tt = torch.Timer() 
    det = net12:forward(pyramid:cuda())--net12:forward(pyramid:cuda())
    tt:stop()

    distributions = unpacker:forward(det, coordinates)
    
    threshold12 = widget.t1.value/10000  
    threshold12Calib = widget.t2.value/10000
    nmsth1 = widget.t3.value/10000  
    threshold48 = widget.t4.value/10000
    threshold48Calib = widget.t5.value/10000
   

   
    if widget.nms.checked then
        useNMS = 1
    else
        useNMS = 0
    end
   
    if widget.gnms.checked then
        useGNMS = 1
    else
        useGNMS = 0
    end
    
    print('threshold12 = '..threshold12..' nmsth1 = '..nmsth1..' threshold12Calib = '..threshold12Calib..' threshold48 = '..threshold48..' threshold48Calib = '..threshold48Calib)
    rawresults = {}
    -- function FFI:
    for i,distribution in ipairs(distributions) do
        parseFFI(distribution, -1, rawresults, scales[i])
    end

    if(#rawresults <1) then
        return    
    end
    
 
    -- (7) clean up results
    detections1 = torch.Tensor(#rawresults,5)
        
    for i,res in ipairs(rawresults) do
        local scale = res[3]
        local x = torch.round(res[1]*network_sub/scale)
        local y = torch.round(res[2]*network_sub/scale)
        local w = torch.round(network_fov[1]/scale)
        local h = torch.round(network_fov[2]/scale)
        detections1[{i,1}] = x
        detections1[{i,2}] = y
        detections1[{i,3}] = x+w
        detections1[{i,4}] = y+h
        detections1[{i,5}] = res[4]
    end

    local detperscale = torch.Tensor(EffNScales)
    local scale_lim = torch.Tensor(EffNScales+1):fill(0)
    if EffNScales>1 then
      for i=1,EffNScales do detperscale[i] = distributions[i]:size(2)*distributions[i]:size(3) end
    end
    detmsc={}
    if EffNScales>1 then
      for i=1,EffNScales do scale_lim[i+1] = detperscale[i]+scale_lim[i] end
      for i=1,EffNScales do detmsc[i]=detections1[{{1+scale_lim[i],scale_lim[i+1]},{}}] end
    elseif EffNScales==1 then
      detmsc[1]=detections1
    end

    detmsc,numberlevels=KeepPositives(detmsc,threshold12)
    
    if #detmsc==0 then
      detections1=nil
      goto skip2end
    end
    detections1=(concatdets(detmsc)):clone()
    
    print("net12 done") 
    
    --nms devrait aller apres calib
    im12c=im:clone()
    im12c=normalize(im12c,stat12calib,ich)
    for r=1,numberlevels do detmsc[r] = applyCalibNet(calib16, im12c, detmsc[r], FilterSize1, threshold12Calib,ich) end
    print("12calib done") 
    detections2=(concatdets(detmsc)):clone()
    if(useNMS==1) and #detmsc>0 then
      for r=1,numberlevels do 
      detmsc[r]=applyNet(net12plain,im12,detmsc[r],network_fov,-1,ich)
      detmsc[r],crap = nms(detmsc[r],nmsth1) 
      end 
       print("nms done") 
    end
    detections3=(concatdets(detmsc)):clone()
    
   

    
    --================Calibration 16 Net===================

    --detections1 = applyCalibNet(calib16, im12c, detections1, FilterSize1, threshold12Calib,ich)

    im48=im:clone()
    im48=normalize(im48,stat48calib,ich)
    detections4=detections3:clone()
    detections4=applyNet(net48,im48,detections4,FilterSize48,threshold48,ich)
    
    print('48net done')
    if detections4==nil then
    detections5=nil
    detections6=nil
    goto skip2end
    end
    if(useGNMS==1) then
       detections5=detections4:clone()
       detections5,crap = nms(detections5,gnmsth)
       print('gnms done')
    end

    im48c=im:clone()
    im48c=normalize(im48c,stat48calib,ich)
    detections6=detections5:clone()
    detections6=applyCalibNet(calib48,im48c,detections6,FilterSize48,threshold48Calib,ich)
    ::skip2end::
    
end



function loadDataFiles(dir)
    local t,i,popen = {},1,io.popen
    for filename in popen('ls -A "'..dir..'"' ):lines() do
        t[i] = dir..filename
        i=i+1
    end
    return t
end



t = loadDataFiles('images/')

network_temp = torch.load(net12_path)
network_temp:evaluate()
print(network_temp)
net12plain=network_temp:clone()
net12plain=network_temp:cuda()

net12 = nn.Sequential()
if string.match(filenameFN, "20") then
    for i = 1, 8 do
      net12:add(network_temp.modules[i])
    end
    classifier1 = nn.Sequential()
    for i = 9, 11 do 
        classifier1:add(network_temp.modules[i])
    end
else
    for i = 1, 5 do
      net12:add(network_temp.modules[i])
    end
    classifier1 = nn.Sequential()
    for i = 6, 8 do 
      classifier1:add(network_temp.modules[i])
    end
end

classifier = nn.SpatialClassifier(classifier1)
net12:add(classifier)
net12:cuda()
print(net12)
print(classifier1)



calib16 = torch.load(calib12_path)
calib16:evaluate()
calib16:cuda()
--calib16:double()
print(calib16)

net48 = torch.load(net48_path)
net48:cuda()
net48:evaluate()
print(net48)
calib48 = torch.load(calib48_path)
calib48:cuda()
calib48:evaluate()
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

-- use a pyramid packer/unpacker
packer = nn.PyramidPacker(net12, scales)
unpacker = nn.PyramidUnPacker(net12)

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
