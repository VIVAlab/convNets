sqlite3 = require "luasql.sqlite3"
require 'torch'
require 'image'
--patterns
s=torch.Tensor{.826446,0.9090909,1,1.1,1.21}--dilation/contraction --
tx=torch.Tensor{-.17,0,.17}--xtranslation proportion
ty=torch.Tensor{-.17,0,.17}--ytranslation proportion
--dimensions ! in number
dims=s:size()[1]--5
dimtx=tx:size()[1]--3
dimty=ty:size()[1]--3
N=dims*dimtx*dimty; --45
sT=torch.Tensor(dims,dimtx,dimty):stride()
--sT=9,3,1
for i=1,N do
    os.execute("mkdir cropped/"..i)
end
os.execute("mkdir ../AFLW_TrainingTest")

function cropImage(iarg,iim) 
  --This function crops an image(CxHxW) with input arguments (x0,y0,xf,yf) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
  --The processing is done in matrix coordinates (i,j).
  local C=1
  local W=1
  local H=1
  if iim:nDimension()==3 then
    C=iim:size(1)
    W=iim:size(3)
    H=iim:size(2)
  elseif iim:nDimension()==2 then
    C=1
    W=iim:size(2)
    H=iim:size(1)
    iim:resize(C,H,W)
  
  elseif iim:nDimension()==4 then
    C=3
    print(4)
    W=iim:size(2)
    H=iim:size(1)
    local iimtemp=torch.Tensor(C,H,W)
    for i=1,3 do
      iimtemp[{i,{},{}}] = torch.add(   torch.add(-iim[{i,{},{}}],1)  ,  torch.cmul(iim[{4,{},{}}] , iim[{i,{},{}}])   )
    end
    iim=iimtemp:clone()
    iimtemp=nil
   
  end
  local oim=torch.Tensor(C,iarg.yf-iarg.y0+1,iarg.xf-iarg.x0+1):fill(0)
  local Mask=torch.ByteTensor(C,iarg.yf-iarg.y0+1,iarg.xf-iarg.x0+1):fill(0)
  local iIxf=math.min(iarg.xf,W)
  local iIx0=math.max(iarg.x0,1)--1 because start index of image=1
  local DiIx=iIxf-iIx0+1 --span of elements in x copyied from input image to output image
  local Offsx=iIx0-math.min(iarg.x0,iIx0) -- index offset of output image in x
  local iIyf=math.min(iarg.yf,H)
  local iIy0=math.max(iarg.y0,1)--1 because start index of image=1
  local DiIy=iIyf-iIy0+1 --span of elements in y copyied from input image to output image
  local Offsy=iIy0-math.min(iarg.y0,iIy0) -- index offset of output image in y

  local Ic=iim:sub(1,-1,iIy0,iIyf,iIx0,iIxf):clone()
  
 Mask:sub(1,-1,Offsy+1,Offsy+DiIy,Offsx+1,Offsx+DiIx):fill(1)

  oim:maskedCopy(Mask,Ic) 
  --oim = image.scale(oim,64,64)
  --oim = image.rgb2y(oim)  --TODO change for grayscale option if C==3 consider cases where C=1 and C =4.
return oim			
end

local env  = sqlite3.sqlite3() --env

local conn = env:connect('aflw.sqlite')

cursor,errorString = conn:execute('select * from TTset')
row = cursor:fetch ({}, "a")
cnt = 0;
while row do
  data,errorString = conn:execute('SELECT * from facerect WHERE face_id='..row.face_id)

  r1 = data:fetch({},"a") --coordinates of annotated image ,i.e. face only.

 

  while r1 do
--if row.file_id=='image00035.jpg' then--'image00097.jpg'
print(row.file_id)

co ={}
co.x=r1.x
co.y=r1.y
co.w=r1.w
co.h=r1.h
    local im = image.load('0/'..row.file_id)  -- entire image 
--print(co.x,co.y,co.w,co.h)
    --folder = 1;
    for i=1,dims do
	for j=1,dimtx do
		for k=1,dimty do
             if filename=='image04416.jpg' then
                  local filename=string.gsub(row.file_id, '.jpg', "")

                  filename=string.gsub(filename, '.png', "")
		  local coimc ={x0=1,y0=1,xf=1,yf=1}
		  coimc.x0=torch.round(co.x+tx[j]*co.w*s[i]-(s[i]-1)*co.w/2)  --COMPUTE the transformation in x1
		  coimc.y0=torch.round(co.y+ty[k]*co.h*s[i]-(s[i]-1)*co.h/2)  --COMPUTE the transformation in y1
		  coimc.xf=torch.round(co.x+tx[j]*co.w*s[i]-(s[i]-1)*co.w/2+co.w*s[i])  --COMPUTE the transformation in x2
		  coimc.yf=torch.round(co.y+ty[k]*co.h*s[i]-(s[i]-1)*co.h/2+co.h*s[i])  --COMPUTE the transformation in y2

	          local imc = cropImage(coimc,im)-- arguments are structure containing {r1={x,y,w,h},im}
		  

		  if (imc~=nil) then
			folder=1+(i-1)*sT[1]+(j-1)*sT[2]+(k-1)*sT[3]
		    image.savePNG('cropped/'..folder..'/'..cnt..'_'..folder..'_'..filename..'_'..i..j..k..'.png',imc)
                    --[[
                    if (i==3 and j==2 and k==2) then
                      image.savePNG('../AFLW_TrainingTest/'..cnt..'_'..filename..'.png',imc)
                    end
                    --]]
		   --   folder=1+(i-1)*sT[1]+(dimtx-j)*sT[2]+(k-1)*sT[3]
		   -- image.savePNG('cropped/'..folder..'/'..cnt..'_'..folder..'_'..i..(dimtx-j+1)..k..'.png',imcfliplr)
			
			
		  end
		  --folder = folder + 1;
		  collectgarbage()
             end
    		end
	end
	
      end
--end   --if row.file_id=='image00097.jpg' then

    cnt = cnt + 1;
    r1 = data:fetch (r1, "a")--fetch new row

end
  -- reusing the table of results
  row = cursor:fetch (row, "a")
  data:close()
end
cursor:close()
conn:close()
env:close()
--end   --if row.file_id=='image00097.jpg' then

    cnt = cnt + 2;
    r1 = data:fetch (r1, "a")--fetch new row

end
  -- reusing the table of results
  row = cursor:fetch (row, "a")
  data:close()
end
cursor:close()
conn:close()
env:close()
