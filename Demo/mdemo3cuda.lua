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
   GUI = 'g2.ui'
end
norm_data={}
ich = 1
  extrastride=1 -- can be 1 or 2
min_size = 45--180
nscales = 100
scratio = 1/math.sqrt(2)
  gray_path='/home/jblan016/FaceDetection/Cascade/GrayCascadeNet'
  rgb_path='/home/jblan016/FaceDetection/Cascade/RGBCascadeNet'
  filename1='12netFC'--'20net3'--'20netNoFC'--'20net2 (copy)'
  filename2='24net2'--'20net3'--'20netNoFC'--'20net2 (copy)'
if ich == 1 then
  net1_dir = gray_path..'/'..filename1..'/results/'
  calib1_dir =gray_path..'/48calibnet/results/' --20calibnet
  net2_dir = gray_path..'/'..filename2..'/results/'
  calib2_dir =gray_path..'/48calibnet/results/' --20calibnet
  net3_dir = gray_path..'/48net2/results/'
  calib3_dir =gray_path..'/48calibnet/results/'
elseif ich ==3 then
  net1_dir   = rgb_path..'/'..filename1..'/results/'
  calib1_dir = rgb_path..'/12calibnet/results/'
  net2_dir   = gray_path..'/'..filename2..'/results/'
  calib2_dir = gray_path..'/24calibnet/results/' --20calibnet
  net3_dir   = rgb_path..'/48net/results/GoodResults/'
  calib3_dir = rgb_path..'/48calibnet/results/'
end
if string.match(filename1, "20") then
    network_fov = {20,20}
    network_sub = 4*extrastride
else
    network_fov = {12,12}
    network_sub =2*extrastride
end
if string.match(calib1_dir, "20") then
    FilterSize1 = {20,20}
elseif string.match(calib1_dir, "12") then
    FilterSize1 = {12,12}
elseif string.match(calib1_dir, "48") then
    FilterSize1 = {48,48}
end


stat1net={}
stat1net.mean={}
stat1net.std={}
stat2net={}
stat2net.mean={}
stat2net.std={}
stat3net={}
stat3net.mean={}
stat3net.std={}
if ich ==3 then
  for i=1,ich do
    stat1net.mean[i]=torch.load(net1_dir..'mean.dat')[i]
    stat1net.std[i]=torch.load(net1_dir..'std.dat')[i]
    stat2net.mean[i]=torch.load(net2_dir..'mean.dat')[i]
    stat2net.std[i]=torch.load(net2_dir..'std.dat')[i]
    stat3net.mean[i]=torch.load(net3_dir..'mean.dat')[i]
    stat3net.std[i]=torch.load(net3_dir..'std.dat')[i]
  end
elseif ich==1 then
  stat1net.mean=torch.load(net1_dir..'mean.dat')[1]
  stat1net.std=torch.load(net1_dir..'std.dat')[1]
  stat2net.mean=torch.load(net2_dir..'mean.dat')[1]
  stat2net.std=torch.load(net2_dir..'std.dat')[1]
  stat3net.mean=torch.load(net3_dir..'mean.dat')[1]
  stat3net.std=torch.load(net3_dir..'std.dat')[1]
end
stat1calib={}
stat1calib.mean={}
stat1calib.std={}
stat2calib={}
stat2calib.mean={}
stat2calib.std={}
stat3calib={}
stat3calib.mean={}
stat3calib.std={}
if ich ==3 then
  for i=1,ich do
    stat1calib.mean[i]=torch.load(calib1_dir..'mean.dat')[i]
    stat1calib.std[i]=torch.load(calib1_dir..'std.dat')[i]
    stat2calib.mean[i]=torch.load(calib2_dir..'mean.dat')[i]
    stat2calib.std[i]=torch.load(calib2_dir..'std.dat')[i]
    stat3calib.mean[i]=torch.load(calib3_dir..'mean.dat')[i]
    stat3calib.std[i]=torch.load(calib3_dir..'std.dat')[i]
  end
elseif ich==1 then
  stat1calib.mean=torch.load(calib1_dir..'mean.dat')[1]
  stat1calib.std=torch.load(calib1_dir..'std.dat')[1]
  stat2calib.mean=torch.load(calib2_dir..'mean.dat')[1]
  stat2calib.std=torch.load(calib2_dir..'std.dat')[1]
  stat3calib.mean=torch.load(calib3_dir..'mean.dat')[1]
  stat3calib.std=torch.load(calib3_dir..'std.dat')[1]
end
  extrastridenames={"",'2'}
  net1_path = net1_dir..'model'..extrastridenames[extrastride]..'.net'
  calib1_path = calib1_dir..'model1.net'
  net2_path = net2_dir..'model.net'
  calib2_path = calib2_dir..'model.net'
  net3_path = net3_dir..'model.net'
  calib3_path = calib3_dir..'model1.net'
--net1_path='/home/jblan016/FaceDetection/Cascade/GrayCascadeNet/1net/results/modelNoBN.net'
imgcnt = 1

FilterSize2={24,24}
FilterSize3 = {48,48}
nmsth1 = .9
nmsth2 = .9
gnmsth = .2
--TODO replace this
threshold2=.03
threshold2Calib=.5
--



scale_min = network_fov[2]/min_size

--[[
function KeepPositives(detmsc,threshold1)
  error('fix ipairs problem to prevent nil element encounter')
  local mask = torch.Tensor()
  local Ndets=0
  for i,v in ipairs(detmsc) do --wrong
  Ndets=Ndets+1
  end
  local newdetmsc={}
  local r=0
  local masksum=0
  for i,v in ipairs(detmsc) do
    mask=v[{{},5}]:ge(threshold1)
    masksum=torch.sum(mask)
    if masksum>0 then
    r=r+1
    newdetmsc[r]=v[torch.repeatTensor(mask,5,1):t()]:clone()
    newdetmsc[r]=newdetmsc[r]:resize((#newdetmsc[r])[1]/5,5)
    end
  end
  return newdetmsc,r
end
--]]
function pyramidscales(n_max,FilterSize,MinFaceSize,sc,imW,imH)  
  --Outputs the number of pyramid levels and the scaling factors given (FilterSize,Minimum Face Size,scaling factor step,image Height,image Width)   
  --scaling factor  0<sc<1  
  local s=0
  local Npl=0
   s=FilterSize/MinFaceSize --initial default scale

  Npl=math.floor(-math.log(math.min(imW,imH)/MinFaceSize)/math.log(sc))+1 --Max Number of Pyramid Levels(Npl) given image width height, initial scaling "s" and scaling level ratio "sc" (smaller than 1)
  Npl=math.min(n_max,Npl)
  local scales={}

  --Fs=torch.Tensor(Npl)
  for k=1,Npl do
    scales[k]=s*(sc)^(Npl-k)
  --  Fs[k]=MinFaceSize/(sc)^(Npl-k)-- Apparent filtersize
  end
  return scales, Npl
end

function verticalExp(Dets)
  local Dy=torch.Tensor()
  Dy=Dets[{{},4}]-Dets[{{},2}]+1
  Dets[{{},2}]=Dets[{{},2}]-.1*Dy
  Dets[{{},4}]=Dets[{{},4}]+.1*Dy
  return Dets
end

function countTable(Table,M)
    local qq=0
    for i=1,M do
        if #(#Table[i])>0 then
          qq=qq+1
        end
    end
    return qq
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
    if widget.s7.checked then
        s7 = 1
    else
        s7 = 0
    end
    if widget.s8.checked then
        s8 = 1
    else
        s8 = 0
    end
    if widget.s9.checked then
        s9 = 1
    else
        s9 = 0
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
    if s7==1 then
        if(detections7~=nil) then
            win:setcolor(1,1,0)
            win:setlinewidth(2)
            for i=1,detections7:size(1) do
                detect=detections7[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s8==1 then
        if(detections8~=nil) then
            win:setcolor(1,0,1)
            win:setlinewidth(2)
            for i=1,detections8:size(1) do
                detect=detections8[i]
                win:rectangle(detect[1], detect[2], detect[3]-detect[1], detect[4]-detect[2])
                win:stroke()
                win:setfont(qt.QFont{serif=false,italic=false,size=16})
                win:moveto(detect[1], detect[2]-1)
                win:show(string.format("%1.2f",detect[5]))
            end
        end
    end
    if s9==1 then
        if(detections9~=nil) then
            win:setcolor(0,1,1)
            win:setlinewidth(2)
            for i=1,detections9:size(1) do
                detect=detections9[i]
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
    detections7 = nil
    detections8 = nil
    detections9 = nil  
end


function concatdets(detsMSc,M)
  
  local Nsc=0
  local Ntot = 0
  local dets = torch.Tensor()
  local v=torch.Tensor()
  local p=0
  local scale_lim = torch.Tensor()

  for i=1,M do
    if #(#detsMSc[i])>0 then
      Nsc=Nsc+1
    end
  end
  if Nsc==0 then
    goto skipconcatdets
  end

  scale_lim = torch.Tensor(Nsc+1):fill(0)
  
  for i=1,M do
    if #(#detsMSc[i])>0 then
      p=p+1
      Ntot=detsMSc[i]:size(1)+Ntot
      scale_lim[p+1] = detsMSc[i]:size(1)+scale_lim[p]
    end
  end
  p=0
  dets = torch.Tensor(Ntot,5)
  
  for i=1,M do
    if #(#detsMSc[i])>0 then
      p=p+1
      dets[{{1+scale_lim[p],scale_lim[p+1]},{}}]=detsMSc[i]:clone()
    end
  end

  ::skipconcatdets::
  return dets
end


function cropImage(iarg, iim, isize)
    -- This function crops an image(CxHxW) with input arguments (x1,y1,x2,y2) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
    -- The processing is done in matrix coordinates (i,j).
    local img_width = iim:size(3)
    local img_hight = iim:size(2)
    local dy=0
    local dx=0
          dy=iarg[4] - iarg[2] + 1
          dx=iarg[3] - iarg[1] + 1
    local oim = torch.Tensor(ich, dy, dx):fill(0)
    local Mask = torch.ByteTensor(ich, dy, dx):fill(0)
    local iIx2 = 0
    local iIx1 = 0
    local DiIx = 0
    local Offsx = 0
    local iIy2 = 0
    local iIy1 = 0
    local DiIy = 0
    local Offsy = 0

     iIx2 = math.min(iarg[3], img_width)
     iIx1 = math.max(iarg[1], 1)  -- 1 because start index of image=1
     DiIx = (iIx2 - iIx1 + 1)  -- span of elements in x copyied from input image to output image
     Offsx = (iIx1 - math.min(iarg[1], iIx1)) -- index offset of output image in x

     iIy2 = math.min(iarg[4], img_hight)
     iIy1 = math.max(iarg[2], 1)  -- 1 because start index of image=1
     DiIy = (iIy2 - iIy1 + 1)  -- span of elements in y copyied from input image to output image
     Offsy = (iIy1 - math.min(iarg[2], iIy1)) -- index offset of output image in y
    --print('iargs')
    --print(iarg[2],iarg[4],iarg[1],iarg[3])
    --print(iim:size())
    --print('iIy1, iIy2, iIx1, iIx2')
    --print(iIy1, iIy2, iIx1, iIx2)
    local Ic=torch.Tensor()
     Ic = iim:sub(1, -1, iIy1, iIy2, iIx1, iIx2):clone()

    Mask:sub(1, -1, Offsy+1, Offsy+DiIy, Offsx+1, Offsx+DiIx):fill(1)

    oim:maskedCopy(Mask, Ic)

    oim = image.scale(oim, isize[1], isize[2])
    
    return oim          
end



function applyNet(model, iimage, detections, isize, threshold,ich)
        local iimVec = torch.Tensor(detections:size(1), ich, isize[2], isize[1])
        detect = torch.Tensor()
        for i = 1, detections:size(1) do
            local detect = detections[{i,{1,4}}]
            iimVec[i] = cropImage(detect, iimage, isize)
            
        end
        local o = torch.Tensor()
        o = torch.exp(model:forward(iimVec:cuda())):double()
        local cnt = 0
        local Mask=torch.ByteTensor()
        local k = torch.LongTensor(o:size(1))
        Mask=(o[{{},1}]):gt(threshold)
        cnt=torch.sum(Mask)
        o = o[{{},1}]
        o = o[Mask]
        local detectionsR = torch.Tensor()
        if(cnt > 0) then
            detectionsR = detections[(Mask:repeatTensor(5,1)):t()]
            detectionsR = detectionsR:reshape(detectionsR:size(1)/5,5):clone()
            detectionsR[{{},5}] = o:clone()
        end
    return detectionsR
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
  --Ivec=nil
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
  local dX=torch.Tensor()
  dX=xmin2-xmax1
  
  local dx=torch.cmax(dX,dswap:fill(0))
  local Xfit=dx:ge(0)
  local ymin2=torch.cmin(D[{{},4}],dswap:fill(d[4]))
  local ymax1=torch.cmax(D[{{},2}],dswap:fill(d[2]))
  local dY=torch.Tensor()
  dY=ymin2-ymax1
  local dy=torch.cmax(dY,dswap:fill(0))
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
    local img=torch.Tensor()
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



function applyCalibNet(model,iim,detections,isize,threshold,ich)
    if(detections~=nil) then
        local iimVec = torch.Tensor(detections:size(1),ich,isize[2],isize[1])
        local detect=torch.Tensor()
        for i=1,detections:size(1) do
            detect=detections[{i,{1,4}}]:clone()
            iimVec[i] = cropImage(detect, iim, isize)
        end
        local ca1 = torch.exp(model:forward(iimVec:cuda())):double()  -- ca1_(#det x 45 )
        local trans = torch.Tensor(ca1:size(1),3):zero()
        local c = 0

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


-- process function
function process()
    -- grab frame
    if camuse==1 then
      frame = camera:forward()
    else
      frame = image.load(t[imgcnt])
    end
    --print(t[imgcnt])
    threshold1 = widget.t1.value/10000  
    threshold1Calib = widget.t2.value/10000
    nmsth1 = widget.t3.value/10000  
    threshold2 = widget.t4.value/10000  
    threshold2Calib = widget.t5.value/10000
    nmsth2 = widget.t6.value/10000 
    threshold3 = widget.t7.value/10000
    gnmsth = widget.t8.value/10000 
    threshold3Calib = widget.t9.value/10000
    -- process input at multiple scales    
    im = frame2im(frame,ich)
    scales,EffNScales=pyramidscales(nscales,network_fov[2],min_size,scratio,frame:size(3),frame:size(2))
    if EffNScales<1 then
    goto skip2end
    end

 
    -- Normalise mean and variance for each channel
    im1=im:clone()
    im1=normalize(im1,stat1net,ich)
    detmsc={}
    local im1res=torch.DoubleTensor()
    local heatmap=torch.CudaTensor()
    local NdetsInScale=0
    tt = torch.Timer() 
    local xindx=torch.DoubleTensor()
    local yindx=torch.DoubleTensor()
    local Mask=torch.ByteTensor()
    local N=0
    local w = 0
    local h = 0
    local r = 0
    local numberlevels=0
    for i=1,EffNScales do
        w = torch.round(network_fov[1]/scales[i])
        h = torch.round(network_fov[2]/scales[i])
        im1res=image.scale(im1,torch.round(scales[i]*frame:size(3)),torch.round(scales[i]*frame:size(2)),'bicubic')
        im1res=im1res:cuda()
        im1res=(nn.Unsqueeze(1):cuda()):forward(im1res)
        heatmap=(app_feat:forward(im1res))
        heatmap=(nn.Squeeze(1):cuda()):forward(heatmap)
        heatmap=classif:forward(heatmap)
        heatmap=torch.exp(heatmap:double())[1]
        Mask=heatmap:ge(threshold1)
        xindx=torch.range(1,heatmap:size(2))-1
        xindx=torch.repeatTensor(xindx,heatmap:size(1),1)
        yindx=torch.range(1,heatmap:size(1))-1
        yindx=torch.repeatTensor(yindx,heatmap:size(2),1)
        yindx=yindx:t()
        NdetsInScale=torch.sum(Mask)
        if NdetsInScale>0 then
          r=r+1
          detmsc[r]=torch.Tensor(NdetsInScale,5)
          detmsc[r][{{},5}]=heatmap[Mask]:clone()
          detmsc[r][{{},1}]=torch.round(xindx[Mask]:clone()*network_sub/scales[i])+1 -- x
          detmsc[r][{{},3}]=(detmsc[r][{{},1}]):clone()+w-1
          detmsc[r][{{},2}]=torch.round(yindx[Mask]:clone()*network_sub/scales[i])+1 -- y
          detmsc[r][{{},4}]=detmsc[r][{{},2}]:clone()+h-1
        end
        numberlevels=r
    end
    tt:stop()

    
   

   

    
    print('threshold1 = '..threshold1..' nmsth1 = '..nmsth1..' threshold1Calib = '..threshold1Calib..' threshold3 = '..threshold3..' threshold3Calib = '..threshold3Calib)
    
    --[[
    local detperscale = torch.Tensor(EffNScales)
    local scale_lim = torch.Tensor(EffNScales+1):fill(0)
    if EffNScales>1 then
      for i=1,EffNScales do detperscale[i] = distributions[i]:size(2)*distributions[i]:size(3) end
    end
    if EffNScales>1 then
      for i=1,EffNScales do scale_lim[i+1] = detperscale[i]+scale_lim[i] end
      for i=1,EffNScales do detmsc[i]=detections1[{{1+scale_lim[i],scale_lim[i+1]},{}}] end
    elseif EffNScales==1 then
      detmsc[1]=detections1
    end
    detmsc,numberlevels=KeepPositives(detmsc,threshold1)
    --]]
    if numberlevels==0 then
      detections1=nil
      detections2=nil
      detections3=nil
      detections4=nil
      detections5=nil
      detections6=nil
      detections7=nil
      detections8=nil
      detections9=nil
      goto skip2end
    end
    detections1=(concatdets(detmsc,numberlevels)):clone()
    
    print("net1 done") 

    im1c=im:clone()
    im1c=normalize(im1c,stat1calib,ich)
    for r=1,numberlevels do detmsc[r] = applyCalibNet(calib1, im1c, detmsc[r], FilterSize1, threshold1Calib,ich) end

    print("calib1 done") 
    detections2=(concatdets(detmsc,numberlevels)):clone()
    if numberlevels>0 then
      for r=1,numberlevels do 
      detmsc[r]=applyNet(net1plain,im1,detmsc[r],network_fov,-1,ich)
      detmsc[r],crap = nms(detmsc[r],nmsth1)
      end 
       print("nms1 done") 
    end
    detections3=(concatdets(detmsc,numberlevels)):clone()
    local rr=0
    local detmsc2={}
    if threshold2>0 then 
      im2=im:clone()
      im2=normalize(im2,stat2net,ich)
      if numberlevels>0 then
        for r=1,numberlevels do 
          detmsc[r]=applyNet(net2,im2,detmsc[r],FilterSize2,threshold2,ich)
          if #(#detmsc[r])>0 then
            rr=rr+1
            detmsc2[rr]=detmsc[r]:clone()
          end
        end 
        numberlevels=rr
      end
      print("net2 done") 
    else
      for r=1,numberlevels do 
        if #(#detmsc[r])>0 then
            rr=rr+1
            detmsc2[rr]=detmsc[r]:clone()
        end
      end 
    print("net2 skipped")
    end 
    --detmsc=nil
    rr=nil
    detections4=(concatdets(detmsc2,numberlevels)):clone()
    
    if numberlevels==0 then 
      detections4=nil
      detections5=nil
      detections6=nil
      detections7=nil
      detections8=nil
      detections9=nil
      print_r(detections3)
      print_r(detmsc2)
      print_r(detmsc)
      pause()
      goto skip2end
    end
    
    
    if threshold2Calib>0 then
    im2c=im:clone()
    im2c=normalize(im2c,stat2calib,ich)
    for r=1,numberlevels do detmsc2[r] = applyCalibNet(calib2, im2c, detmsc2[r], FilterSize2, threshold2Calib,ich) end
    print("calib2 done") 
    else
    print("calib2 skipped")
    end 
    detections5=(concatdets(detmsc2,numberlevels)):clone()
   if nmsth2<1 then
     if numberlevels>0 then
        for r=1,numberlevels do 
        detmsc2[r]=applyNet(net2,im2,detmsc2[r],FilterSize2,-1,ich)
        detmsc2[r],crap = nms(detmsc2[r],nmsth2)
        end 
         print("nms2 done") 
      end
   else
     print("nms2 skipped")    
   end
    detections6=(concatdets(detmsc2,numberlevels)):clone()
    
    im3=im:clone()
    im3=normalize(im3,stat3calib,ich)
    detections7=detections6:clone()
    detections7=applyNet(net3,im3,detections7,FilterSize3,threshold3,ich)
    print('3net done')
    if #(#(detections7))==0 then
      detections7=nil
      detections8=nil
      detections9=nil
      goto skip2end
    end
    detections8=detections7:clone()
    detections8,crap = nms(detections8,gnmsth)
    print('gnms done')
   
    im3c=im:clone()
    im3c=normalize(im3c,stat3calib,ich)
    detections9=detections8:clone()
    detections9=applyCalibNet(calib3,im3c,detections9,FilterSize3,threshold3Calib,ich)
    print('calib3 done')
    --detections9=verticalExp(detections9)
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

app_feat = torch.load(net1_path)
Nmods=#app_feat.modules

app_feat:evaluate()
print('cnet1 filter')
print(app_feat)
app_feat=app_feat:cuda()
net1plain=app_feat:clone()
classif=nn.Sequential()

classif=nn.Sequential()
for i=1,3 do
classif:add(app_feat.modules[Nmods-2])
app_feat:remove(Nmods-2)
end
classif=nn.SpatialClassifier(classif)
classif=classif:cuda()
calib1 = torch.load(calib1_path)
calib1:evaluate()
calib1:cuda()
--calib1:double()
print('calib1 filter')
print(calib1)
net2 = torch.load(net2_path) --TODO:change notation net1->net1 calib1->calib1 etc...
net2:cuda()
net2:evaluate()
calib2 = torch.load(calib2_path)
calib2:cuda()
calib2:evaluate()
net3 = torch.load(net3_path)
net3:cuda()
net3:evaluate()
print(net3)
calib3 = torch.load(calib3_path)
calib3:cuda()
calib3:evaluate()
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


-- setup GUI (external UI file)
widget = qtuiloader.load('g2.ui')
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
