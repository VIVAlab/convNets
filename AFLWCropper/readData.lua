sqlite3 = require "luasql.sqlite3"
require 'torch'
require 'image'

local env  = sqlite3.sqlite3()

local conn = env:connect('aflw.sqlite')

cursor,errorString = conn:execute('select * from faces')
row = cursor:fetch ({}, "a")
cnt = 0;
while row do
  data,errorString = conn:execute('SELECT * from facerect WHERE face_id='..row.face_id)
  r1 = data:fetch({},"a")
  while r1 do
    print(row.file_id)
    im = image.load('0/'..row.file_id)
    
	r = {x1 = r1.x+1, y1 = r1.y+1,x2 = r1.x + r1.w, y2 = r1.y+r1.h}
	rc ={x1 = 1, y1 = 1, x2 = r1.w , y2 = r1.h}
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
			rc.x2 = r1.w - r.x2 + im:size(3)
			r.x2 = im:size(3)
			
		end
		if(r.y2 > im:size(2)) then
			rc.y2 = r1.h- r.y2 + im:size(2)
			r.y2 = im:size(2)
			
		end
	if(r.x1<r.x2 and r.y1<r.y2) then
		
		im = image.crop(im, r.x1, r.y1, r.x2,r.y2)
		imc = im:clone()
		if(rc.x1>1) then
			--print('s0 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
			imc = torch.cat(torch.zeros(imc:size(1),imc:size(2),rc.x1-1),imc,3)
		end
		if(rc.x2<r1.w) then
			--print('s1 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
			imc = torch.cat(imc,torch.zeros(imc:size(1),imc:size(2),r1.w - rc.x2),3)
		end
		if(rc.y1>1) then
			--print('s2 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
			imc = torch.cat(torch.zeros(imc:size(1),rc.y1-1,imc:size(3)),imc,2)
		end
		if(rc.y2<r1.h) then
			--print('s3 '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
			imc = torch.cat(imc,torch.zeros(imc:size(1),r1.h - rc.y2,imc:size(3)),2)
		
		end
		--print('st '..r.x1..' '..r.y1..' '..r.x2 ..' '..r.y2..' '..rc.x1..' '..rc.y1..' '..rc.x2..' '..rc.y2..' '..im:size(2)..' '..im:size(3)..' '..imc:size(2)..' '..imc:size(3))
		--imc:narrow(2,rc.y1,rc.y2):narrow(3,rc.x1,rc.x2):copy(im)
		if(imc:size(2)~=imc:size(3)) then
			print("problem in "..row.file_id.." and "..row.face_id)
		end
		imc = image.scale(imc,64,64)
		image.savePNG('cropped/'..cnt..'.png',imc)
		cnt = cnt + 1;
	end
	r1 = data:fetch (r1, "a")
  end
  -- reusing the table of results
  row = cursor:fetch (row, "a")
  data:close() 	
end

cursor:close()
conn:close()
env:close()
