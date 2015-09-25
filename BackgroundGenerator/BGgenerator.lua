require 'torch'
require 'nnx'
require 'cunn'
require 'image'
require 'PyramidPacker'
require 'PyramidUnPacker'


torch.setdefaulttensortype('torch.FloatTensor')
threshold = 0.5

os.execute('mkdir NegativeData')

function loadDataFiles(dir)
    local i, t,l, popen = 0, {},{}, io.popen  
    for filename in popen('ls -A "'..dir..'"' ):lines() do
	   i = i + 1
       t[i] = dir..filename
    end
    return t
end

network1 = torch.load('model.net')
network = nn.Sequential()
for i=1,5 do
	network:add(network1.modules[i])
end
classifier1 = nn.Sequential()
for i=6,8 do 
	classifier1:add(network1.modules[i])
end
classifier = nn.SpatialClassifier(classifier1)
network:add(classifier)
network:cuda()


scales = {0.2}

packer = nn.PyramidPacker(network, scales)
unpacker = nn.PyramidUnPacker(network)

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
          table.insert(blobs,entry)
      end
    end
  end
end

network_fov = 12
network_sub = 2
cnt=0;

-- process function
function process(imAdd)
   -- grab frame
   print(imAdd)
   frame = image.load(imAdd)
  
   im = image.rgb2yuv(frame):float()
   local fmean = im:mean()
   local fstd = im:std()
   im:add(-fmean)
   im:div(fstd):cuda()
   pyramid, coordinates = packer:forward(im)	
   
   det = network:forward(pyramid:cuda())
   
   distributions = unpacker:forward(det, coordinates)
   

   rawresults = {}
   -- function FFI:
   for i,distribution in ipairs(distributions) do
       parseFFI(distribution, threshold, rawresults, scales[i])
   end
   -- (7) clean up results
   detections = {}
   for i,res in ipairs(rawresults) do
      local scale = res[3]
      local x = res[1]*network_sub/scale
      local y = res[2]*network_sub/scale
      local w = network_fov/scale
      local h = network_fov/scale
      detections[i] = {x=x, y=y, w=w, h=h}
   end
   for i=1,#detections do
		im2 = image.crop(frame,detections[i].x,detections[i].y,detections[i].x+detections[i].w,detections[i].y+detections[i].h)
		im2 = image.scale(im2,64,64)
		image.save('NegativeData/i'..cnt..'.jpg',im2)
		cnt =cnt+1
   end
end

address = loadDataFiles('../dataset/negative/')
for i=1,#address do
	process(address[i])
end
