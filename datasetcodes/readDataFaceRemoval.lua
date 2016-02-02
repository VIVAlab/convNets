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
  
  print(row.file_id)
  while r1 do
    
        im = image.load('images/'..row.file_id)
    
	r = {x = torch.floor(r1.x+1-0.1*r1.w), y = torch.floor(r1.y+1-0.1*r1.h),w = torch.floor(1.2*r1.w), h = torch.floor(1.2*r1.h)}
	--rc ={x1 = 1, y1 = 1, x2 = r1.w , y2 = r1.h}
                if(r.x < 1) then
			r.x = 1
			--rc.x1 = v + 1
		end
		if(r.y < 1) then
			r.y = 1
			--rc.y1 = v + 1
		end
		if(r.x + r.w > im:size(3)) then
			--rc.x2 = r1.w - r.x2 + im:size(3)
			r.w = im:size(3) - r.x
			
		end
		if(r.y + r.h > im:size(2)) then
			--rc.y2 = r1.h- r.y2 + im:size(2)
			r.h = im:size(2) - r.y
			
		end
	--if(r.x<r.x+r.w and r.y<r.y+r.h) then
		
		im:narrow(3,r.x,r.w):narrow(2,r.y,r.h):zero()
		image.save('images/'..row.file_id,im)
		image.save('imagesCopy/'..row.file_id,im)
	--end
	r1 = data:fetch (r1, "a")
	collectgarbage()
  end
  
  cnt = cnt + 1;
  -- reusing the table of results
  row = cursor:fetch (row, "a")
  data:close() 	
end

cursor:close()
conn:close()
env:close()
