import tensorflow as tf
import numpy as np
from scipy import misc
from scipy import interpolate
import skimage.transform as skt
import math
import copy
import time
import skimage.util as sku
import cv2
from os import listdir
from os.path import isfile, join

import TFNets

#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

camuse=0
#if camuse==1:				#NOTE: Not converted to python
  #camera = image.Camera(0)
  #GUI = 'g2.ui'

ich = 1
extrastride = 1 #NOTE: can be 1 or 2. TF implementation needs additional tests of extrastride.
min_size = 35 #180
nscales = 100

scratio = 1/math.sqrt(2)
gray_path='/home/danlaptop/FaceDetection/Cascade/TFNets/'
rgb_path='/home/danlaptop/FaceDetection/Cascade/'
image_path = '/home/danlaptop/FaceDetection/Cascade/images_2/'
#result_path = '/home/danlaptop/FaceDetection/Cascade/results/' #Used when we want to save the results to file

net1name = '20net'
net2name = '20net'
net3name = '48net'
calib1name = '12cnet'
calib2name = '12cnet'
calib3name = '48cnet'

threshold1	= 0.15
threshold1Calib	= 0.4
nmsth1		= 0.95
threshold2	= 0
threshold2Calib	= 0
nmsth2		= 1
threshold3	= 0.5
gnmsth		= 0.2
threshold3Calib = 0.8

if ich == 1:
  net1_dir = gray_path
  calib1_dir = gray_path
  net2_dir = gray_path
  calib2_dir = gray_path
  net3_dir = gray_path
  calib3_dir = gray_path
elif ich == 3:
  net1_dir   = rgb_path
  calib1_dir = rgb_path
  net2_dir   = gray_path
  calib2_dir = gray_path
  net3_dir   = rgb_path
  calib3_dir = rgb_path

if '20' in net1name:
  net1_fov = [20,20]
  network_sub = 4*extrastride
elif '12' in net1name:
  net1_fov = [12,12]
  network_sub =2*extrastride

if '20' in net2name:
  net2_fov = [20,20]
elif '12' in net2name:
  net2_fov = [12,12]
elif '48' in net2name:
  net2_fov = [48,48]
elif '24' in net2name:
  net2_fov = [24,24]

if '20' in net3name:
  net3_fov = [20,20]
elif '12' in net3name:
  net3_fov = [12,12]
elif '48' in net3name:
  net3_fov = [48,48]
elif '24' in net3name:
  net3_fov = [24,24]
	
if '20' in calib1name:
  calib1_fov = [20,20]
elif '12' in calib1name:
  calib1_fov = [12,12]
elif '48' in calib1name:
  calib1_fov = [48,48]

if '20' in calib2name:
  calib2_fov = [20,20]
elif '12' in calib2name:
  calib2_fov = [12,12]
elif '48' in calib2name:
  calib2_fov = [48,48]
elif '24' in calib2name:
  calib2_fov = [24,24]

if '20' in calib3name:
  calib3_fov = [20,20]
elif '12' in calib3name:
  calib3_fov = [12,12]
elif '48' in calib3name:
  calib3_fov = [48,48]
elif '24' in calib3name:
  calib3_fov = [24,24]

#Will need to test/change the loading of mean and std values when ich=3
if ich==1:
  stat1net_mean	= np.loadtxt(net1_dir + 'mean_20net.txt')[1]
  stat1net_std	= np.loadtxt(net1_dir + 'std_20net.txt')[1]
  stat2net_mean	= np.loadtxt(net2_dir + 'mean_20net.txt')[1]
  stat2net_std	= np.loadtxt(net2_dir + 'std_20net.txt')[1]
  stat3net_mean	= np.loadtxt(net3_dir + 'mean_48net.txt')[1]
  stat3net_std	= np.loadtxt(net3_dir + 'std_48net.txt')[1]
elif ich ==3:
  stat1net_mean = []
  stat1net_std = []
  stat2net_mean = []
  stat2net_std = []
  stat3net_mean = []
  stat3net_std = []
  for i in range(ich):
    stat1net_mean.append= np.loadtxt(net1_dir + 'mean_20net.txt')[i]
    stat1net_std.append	= np.loadtxt(net1_dir + 'std_20net.txt')[i]
    stat2net_mean.append= np.loadtxt(net2_dir + 'mean_20net.txt')[i]
    stat2net_std.append	= np.loadtxt(net2_dir + 'std_20net.txt')[i]
    stat3net_mean.append= np.loadtxt(net3_dir + 'mean_48net.txt')[i]
    stat3net_std.append	= np.loadtxt(net3_dir + 'std_48net.txt')[i]

if ich==1:
  stat1calib_mean= np.loadtxt(calib1_dir + 'mean_12cnet.txt')[1]
  stat1calib_std = np.loadtxt(calib1_dir + 'std_12cnet.txt')[1]
  stat2calib_mean= np.loadtxt(calib2_dir + 'mean_12cnet.txt')[1]
  stat2calib_std = np.loadtxt(calib2_dir + 'std_12cnet.txt')[1]
  stat3calib_mean= np.loadtxt(calib3_dir + 'mean_48cnet.txt')[1]
  stat3calib_std = np.loadtxt(calib3_dir + 'std_48cnet.txt')[1]
elif ich == 3:
  stat1calib_mean= []
  stat1calib_std = []
  stat2calib_mean= []
  stat2calib_std = []
  stat3calib_mean= []
  stat3calib_std = []
  for i in range(ich):
    stat1calib_mean.append=np.loadtxt(calib1_dir + 'mean_12cnet.txt')[i]
    stat1calib_std.append =np.loadtxt(calib1_dir + 'std_12cnet.txt')[i]
    stat2calib_mean.append=np.loadtxt(calib2_dir + 'mean_12cnet.txt')[i]
    stat2calib_std.append =np.loadtxt(calib2_dir + 'std_12cnet.txt')[i]
    stat3calib_mean.append=np.loadtxt(calib3_dir + 'mean_48cnet.txt')[i]
    stat3calib_std.append =np.loadtxt(calib3_dir + 'std_48cnet.txt')[i]

net1_path	= net1_dir	+ 'model_' + net1name	+ '.py'
calib1_path	= calib1_dir	+ 'model_' + calib1name + '.py'
net2_path	= net2_dir	+ 'model_' + net2name	+ '.py'
calib2_path	= calib2_dir	+ 'model_' + calib2name + '.py'
net3_path	= net3_dir	+ 'model_' + net3name	+ '.py'
calib3_path	= calib3_dir	+ 'model_' + calib3name + '.py'

net1_data_path 	= net1_dir 	+ 'model_' + net1name 	+ '_data.npy'
calib1_data_path= calib1_dir 	+ 'model_' + calib1name + '_data.npy'
net2_data_path 	= net2_dir 	+ 'model_' + net2name 	+ '_data.npy'
calib2_data_path= calib2_dir 	+ 'model_' + calib2name + '_data.npy'
net3_data_path 	= net3_dir 	+ 'model_' + net3name 	+ '_data.npy'
calib3_data_path= calib3_dir 	+ 'model_' + calib3name + '_data.npy'

scale_min = net1_fov[1]/min_size

def pyramidscales(n_max,FilterSize,MinFaceSize,sc,imW,imH):
  #Outputs the number of pyramid levels and the scaling factors given (FilterSize,Minimum Face Size,scaling factor step,image Height,image Width)   
  #scaling factor  0<sc<1  
  s=float(FilterSize)/MinFaceSize #initial default scale

  Npl=int(math.floor(-math.log(min(imW,imH)/MinFaceSize)/math.log(sc))+1) #Max Number of Pyramid Levels(Npl) given image width height, initial scaling "s" and scaling level ratio "sc" (smaller than 1)
  Npl=min(n_max,Npl)
  scales=[]

  #Fs=torch.Tensor(Npl)
  for k in range(Npl):
    scales.append(s*(sc**(Npl-(k+1))))
    #Fs[k]=MinFaceSize/(sc)**(Npl-(k+1))# Apparent filtersize
  return scales, Npl

def verticalExp(Dets):
  Dy=Dets[:,3]-Dets[:,1]+1
  Dets[:,1]=Dets[:,1]-.1*Dy
  Dets[:,3]=Dets[:,3]+.1*Dy
  return Dets

def countTable(Table,M):
  qq=0
  for i in range(M):
    if (len(Table[i]))>0:
      qq=qq+1
  return qq

def concatdets(detsMSc,M):
  Nsc=0
  Ntot = 0
  dets = np.array(())
  p=0
  for i in range(M):
    if (len(detsMSc[i]))>0:
      Nsc=Nsc+1
  if Nsc==0:
    return dets

  scale_lim = [0]
  for i in range(M):
    if (len(detsMSc[i]))>0:
      p=p+1
      Ntot=detsMSc[i].shape[0]+Ntot
      scale_lim.append(detsMSc[i].shape[0]+scale_lim[p-1])
  p=0
  dets = np.array(np.zeros((Ntot,5)))
  
  for i in range(M):
    if (len(detsMSc[i]))>0:
      p=p+1
      dets[(scale_lim[p-1]):(scale_lim[p]),:] = copy.deepcopy(detsMSc[i])

  return dets

def cropImage(iarg, iim, isize):
  # This function crops an image(HxWxC) with input arguments (x1,y1,x2,y2) from matrix coordinates (x=j,y=i). (start at (i,j)=(1,1))
  # The processing is done in matrix coordinates (i,j).
  #time_temp = time.time()
  #print("1", time_temp)
  img_width = iim.shape[1]
  img_hight = iim.shape[0]
  dy=iarg[3] - iarg[1] + 1
  dx=iarg[2] - iarg[0] + 1
  Mask = np.zeros((dy, dx, ich), dtype=bool)

  iIx2 = min(iarg[2], img_width-1)
  iIx1 = max(iarg[0], 0)  # 0 because start index of image=0
  DiIx = (iIx2 - iIx1 + 1)  # span of elements in x copied from input image to output image
  Offsx = (iIx1 - min(iarg[0], iIx1)) # index offset of output image in x
  #print("2", time.time() - time_temp)

  iIy2 = min(iarg[3], img_hight-1)
  iIy1 = max(iarg[1], 0)  # 0 because start index of image=0
  DiIy = (iIy2 - iIy1 + 1)  # span of elements in y copied from input image to output image
  Offsy = (iIy1 - min(iarg[1], iIy1)) # index offset of output image in y
  #print('iargs')
  #print(iarg[2],iarg[4],iarg[1],iarg[3])
  #print(iim:size())
  #print('iIy1, iIy2, iIx1, iIx2')
  #print(iIy1, iIy2, iIx1, iIx2)
  Ic = iim[iIy1:(iIy2+1),iIx1:(iIx2+1),:]
  #print("3", time.time() - time_temp)
  
  (Mask[Offsy:Offsy+DiIy, Offsx:Offsx+DiIx,:]).fill(1)
  #print("4", time.time() - time_temp)
  oim = np.zeros((Mask.shape))
  np.putmask(oim, Mask==1, np.reshape(Ic, -1))
  #print("5", time.time() - time_temp)
  
  oim = cv2.resize(oim, (isize[0], isize[1]))
  oim = np.expand_dims(oim, axis=2)
  #print("6", time.time() - time_temp)
  
  return oim

def applyNet(model, input_node, iimage, detections, isize, threshold, ich):
  iimVec = np.zeros((detections.shape[0], isize[1], isize[0], ich))
  for i in range(detections.shape[0]):
    iimVec[i] = cropImage(detections[i][0:4], iimage, isize)

  o = sess.run(model.get_output(), feed_dict={input_node: iimVec})
  cnt = 0
  k = np.zeros((o.shape[0]))
  Mask=(o[:,0]>(threshold))
  cnt=np.sum(Mask)
  o = o[:,0]
  o = o[Mask]
  detectionsR = np.array(())
  if(cnt > 0):
    detectionsR = detections[np.transpose(np.tile(Mask, (5,1)))]
    detectionsR = np.reshape(detectionsR, [detectionsR.shape[0]/5,5])
    detectionsR[:,4] = copy.deepcopy(o)
  return detectionsR

def IoU(d,D): #compare ioU of d with D. d is a row vector and D are a row ordered list of bounding box coordinates
  N=D.shape[0]
  InoU = np.zeros((N))
  O = np.zeros((N))
  dswap = np.zeros((N))
  A=0

  dswap.fill(d[2])
  xmin2 = np.minimum(D[:,2], dswap)
  dswap.fill(d[0])
  xmax1 = np.maximum(D[:,0], dswap)
  dX = xmin2-xmax1+1
  dswap.fill(0)
  dx = np.maximum(dX, dswap)
  Xfit = (dx>=0)

  dswap.fill(d[3])
  ymin2 = np.minimum(D[:,3], dswap)
  dswap.fill(d[1])
  ymax1 = np.maximum(D[:,1], dswap)
  dY = ymin2-ymax1+1
  dy = np.maximum(dY, dswap.fill(0))
  Yfit = (dy>=0)
  XYfit = np.multiply(Xfit,Yfit)	#Xfit:eq(Yfit)  #XYfit =(Xfit AND Yfit)
  #compute intersection area
  InoU[XYfit]=np.multiply(dx[XYfit],dy[XYfit]) 
  #compute overlap
  Temp1 = D[:,2]
  Temp2 = D[:,0]
  Temp3 = D[:,3]
  Temp4 = D[:,1]
  O[XYfit] = np.multiply(Temp1[XYfit]-Temp2[XYfit]+1,Temp3[XYfit]-Temp4[XYfit]+1) #compute only overlaps for non-zero intersections
  O[XYfit] = np.add(O[XYfit],-InoU[XYfit])#compute only overlaps for non-zero intersections
  A=(d[2]-d[0]+1)*(d[3]-d[1]+1)
  O[XYfit] = np.add(O[XYfit],A)  #compute only overlaps for non-zero intersections

  InoU[XYfit]=np.divide(InoU[XYfit],O[XYfit])
  return InoU, XYfit

def nms(boxes, overlap):
  Dets = copy.deepcopy(boxes[:,0:4])
  s = copy.deepcopy(boxes[:,4])
  i = np.argsort(s, axis=0)[::-1]
  s = np.sort(s, axis=0)[::-1]
  N=Dets.shape[0]
  for k in range(N):
    Dets[k]=boxes[i[k]][0:4]
  cnt=0
  i = np.array(range(1,(N+1)))
  while N>1:
    cnt=cnt+1
    IOU, IOUgt0 = IoU(Dets[cnt-1,:],Dets[cnt:,:])
    Mask = np.ones([Dets[cnt:,:].shape[0]], dtype=bool)
    Mask[IOUgt0]=(IOU[IOUgt0]<=overlap)	
    N=np.sum(Mask)
    if N==0:
      s=s[:cnt]
      i=i[:cnt]
      N=0
      Dets=Dets[0:cnt,:]
    else:
      Temp1 = s[cnt:]
      Temp2 = i[cnt:]
      s = np.concatenate((s[:cnt],Temp1[Mask]), axis=0)
      i = np.concatenate((i[:cnt],Temp2[Mask]), axis=0)
      Mask = np.transpose(np.tile(Mask, (4,1)))
      Temp3 = Dets[cnt:,:]
      Dets = np.concatenate((Dets[:cnt,:], np.resize(Temp3[Mask], [N,4])), axis=0)
  if len(s.shape) == 1:
    s = np.expand_dims(s, axis=1)

  return np.concatenate((Dets, s), axis=1), Dets.shape[0]

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def frame2im(fr,ich):
  if ich==1:
    if fr.shape[2]==1:
      img = fr
    elif fr.shape[2]==4:
      img = fr[0:3,:,:]
      img = rgb2gray(img)
    elif fr.shape[2]==3:
      img = rgb2gray(fr)
  elif ich==3:
    if len(fr)==4:
      img = fr[0:3,:,:]
    elif len(fr) == 1:	#Else if statement is untested
      img = np.tile(fr,(ich,1,1))
    elif len(fr)==3:
      img=fr
  return img

def normalize(im,stat_mean, stat_std,ich):
  if ich==1:
    im = np.add(im, -stat_mean)
    im = np.divide(im, stat_std)
  elif ich==3:
    for i in range(ich):
      im[i] = np.add(im, -stat_mean[i])
      im[i] = np.divide(im, stat_std[i])
  return im

def applyCalibNet(model, input, iim,detections,isize,threshold,ich):
  #print("applyCalibNet")
  #time_temp = time.time()
  #print("1", time_temp)
  if (len(detections))>0:
    iimVec = np.zeros((detections.shape[0], isize[1], isize[0], ich))
    #print("2", time.time() - time_temp)

    for i in range(detections.shape[0]):
      iimVec[i] = cropImage(detections[i][0:4], iim, isize)
    #print("3", time.time() - time_temp)
	  
    ca1 = sess.run(model.get_output(), feed_dict={input: iimVec})  # ca1_(#det x 45 )
    #print("4", time.time() - time_temp)
    trans = np.zeros((ca1.shape[0], 3))
    maxlbl = 0
    for i in range(ca1.shape[0]):# for each detection window
      #time_temp = time.time()
      #print("1", time.time() - time_temp)
      c = 0
      maxlbl = np.amax(ca1[i])
      #print("2", time.time() - time_temp)
      for j in range(ca1.shape[1]):	 # for all 45 labels
        if(ca1[i,j] > threshold):
          trans[i] = trans[i] + np.dot(ca1[i,j], T[j])	#trans[i] + T[j]
          c = c + ca1[i,j]						#c + 1
      #print("3", time.time() - time_temp)
      if (c != 0):	 #TODO pick a better way of choosing max
        trans[i,0] = trans[i,0]/c
        trans[i,1] = trans[i,1]/c
        trans[i,2] = trans[i,2]/c
      elif (c == 0):
        for j in range(ca1.shape[1]):
          if(ca1[i,j] == maxlbl):
            trans[i] = trans[i] + np.dot(ca1[i][j], T[j])	#trans[i] + T[j]
            c = c + ca1[i][j]						#c + 1
        trans[i,0] = trans[i,0]/c
        trans[i,1] = trans[i,1]/c
        trans[i,2] = trans[i,2]/c
      #print("4", time.time() - time_temp)
   
    #print("5", time.time() - time_temp)
    for i in range (trans.shape[0]):
      w = abs(detections[i,2]-detections[i,0])+1	#w = detections[i][3]-detections[i][1]  
      h = abs(detections[i,3]-detections[i,1])+1	#h = detections[i][4]-detections[i][2]
      sn = trans[i,0]
      xn = trans[i,1]
      yn = trans[i,2]
      x1 = np.around(detections[i,0] - w*xn+(sn-1)*w/2/sn)
      y1 = np.around(detections[i,1] - w*yn+(sn-1)*h/2/sn)
      x2 = np.around(detections[i,0] - w*xn+(sn-1)*w/2/sn+w/sn)
      y2 = np.around(detections[i,1] - w*yn+(sn-1)*h/2/sn+h/sn)
      detections[i,0] = x1
      detections[i,1] = y1
      detections[i,2] = x2
      detections[i,3] = y2

  #print("6", time.time() - time_temp)
  return detections

# process function
def process():
  frame = misc.imread(image)

  # process input at multiple scales
  im = (frame2im(frame,ich))/255	#The division by 255 is required because of the way torch loaded the images
  scales, EffNScales = pyramidscales(nscales,net1_fov[1],min_size,scratio,frame.shape[1],frame.shape[0])
  if EffNScales<1:
    return True

  # Normalise mean and variance for each channel
  im1=copy.deepcopy(im)
  im = np.expand_dims(im1, axis=3)
  #print("Original")
  #print(im1)
  im1=normalize(im1,stat1net_mean, stat1net_std,ich)
  #print("Normalized")
  #print(im1)
  detmsc=[]
  
  tt = time.time()
  N=0
  r = 0
  numberlevels=0
  for i in range(EffNScales):
    w = np.around(net1_fov[0]/scales[i])
    h = np.around(net1_fov[1]/scales[i])

    #im1res = cv2.resize(np.array((im1), dtype=np.float64), (int(np.around(scales[i]*frame.shape[1])), int(np.around(scales[i]*frame.shape[0]))), interpolation=cv2.INTER_CUBIC)

    im1res = skt.resize(im1, [np.around(scales[i]*frame.shape[1]), np.around(scales[i]*frame.shape[0])], order=3)	#NOTE: order=3 for bicubic
    #print("Rescaled", scales[i])
    #print(im1res)
    #print(im1res.shape)
    im1res = np.expand_dims(im1res, axis=3)
    im1res = np.expand_dims(im1res, axis=0)
    heatmap = sess.run(app_feat.get_output(), feed_dict={app_feat_input: im1res})
    temp_shape = heatmap.shape
    heatmap = sess.run(classif.get_output(), feed_dict={classif_input: heatmap})
    heatmap = heatmap[:,0]
    heatmap = heatmap.reshape(temp_shape[2], temp_shape[1])
	
    Mask = (heatmap >= threshold1)
    xindx = range(0, heatmap.shape[1])
    xindx = np.tile(xindx,(heatmap.shape[0],1))
    yindx = range(0, heatmap.shape[0])
    yindx = np.transpose(np.tile(yindx,(heatmap.shape[1],1)))

    NdetsInScale = np.sum(Mask)
    if NdetsInScale>0:
      r=r+1
      detmsc.append(np.zeros((NdetsInScale, 5)))
      tempArray = detmsc[r-1]
      tempArray[:,4] = copy.deepcopy(heatmap[Mask])
      tempArray[:,0] = np.around(copy.deepcopy(xindx[Mask])*network_sub/scales[i]) +1 # x
      tempArray[:,2] = copy.deepcopy(tempArray[:,0]) +w-1
      tempArray[:,1] = np.around(copy.deepcopy(yindx[Mask])*network_sub/scales[i]) +1 # y
      tempArray[:,3] = copy.deepcopy(tempArray[:,1]) +h-1
    numberlevels=r
  
  print('threshold1 = %.2f' %(threshold1))
  print('threshold1Calib = %.2f' %(threshold1Calib))
  print('nmsth1 = %.2f' %(nmsth1))
  #print('threshold2 = %.2f' %(threshold2))
  #print('threshold2Calib = %.2f' %(threshold2Calib))
  #print('nmsth2 = %.2f', %(nmsth2))
  print('threshold3 = %.2f' %(threshold3))
  print('gnmsth = %.2f' %(gnmsth))
  print('threshold3Calib = %.2f' %(threshold3Calib))


  if numberlevels==0:
    detections1=[]
    detections2=[]
    detections3=[]
    detections4=[]
    detections5=[]
    detections6=[]
    detections7=[]
    detections8=[]
    detections9=[]
    print(len(detections1),len(detections2),len(detections3),len(detections6),len(detections7),len(detections8),len(detections9))
    return detections1
  detections1=concatdets(detmsc,numberlevels)
    
  print("net1 done")
  print(time.time() - tt)

  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections1
  #print(len(detections),len(detections[0]))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)

  im1c=copy.deepcopy(im)
  im1c=normalize(im1c,stat1calib_mean, stat1calib_std,ich)
  for r in range(numberlevels):
    detmsc[r] = applyCalibNet(calib1, calib1_input, im1c, detmsc[r], calib1_fov, threshold1Calib, ich)

  print("calib1 done") 
  print(time.time() - tt)
  detections2=concatdets(detmsc,numberlevels)

  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections2
  #print(len(detections))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)

  if numberlevels>0:
    for r in range(numberlevels):
      detmsc[r],crap = nms(detmsc[r],nmsth1)
    print("nms1 done")
    print(time.time() - tt)
  detections3=concatdets(detmsc,numberlevels)
  rr=0
  detmsc2=[]

  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections3
  #print(len(detections))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)

  if threshold2>0:
    im2=copy.deepcopy(im)
    im2=normalize(im2,stat2net_mean, stat2net_std,ich)
    if numberlevels>0:
      for r in range(numberlevels):
        detmsc[r]=applyNet(net2, net2_input,im2,detmsc[r],net2_fov,threshold2,ich)
        if (len(detmsc[r]))>0:
          rr=rr+1
          detmsc2.append(detmsc[r])
      numberlevels=rr
    print("net2 done") 
    print(time.time() - tt)
  else:
    for r in range(numberlevels):
      if (len(detmsc[r]))>0:
          rr=rr+1
          detmsc2.append(detmsc[r])
    print("net2 skipped")
    print(time.time() - tt)
  if numberlevels==0:
    detections4=[]
    detections5=[]
    detections6=[]
    detections7=[]
    detections8=[]
    detections9=[]
    print(len(detections1),len(detections2),len(detections3),len(detections6),len(detections7),len(detections8),len(detections9))
    return detections3

  detections4=concatdets(detmsc2,numberlevels)
    
    
  if threshold2Calib>0:
    im2c=copy.deepcopy(im)
    im2c=normalize(im2c, stat2calib_mean, stat2calib_std, ich)
    for r in range(numberlevels):
      detmsc2[r] = applyCalibNet(calib2, calib2_input, im2c, detmsc2[r], calib2_fov, threshold2Calib,ich)
    print("calib2 done") 
  else:
    print("calib2 skipped")
  print(time.time() - tt)
  detections5=concatdets(detmsc2,numberlevels)
  if nmsth2<1:
    if numberlevels>0:
      for r in range(numberlevels):
        detmsc2[r],crap = nms(detmsc2[r],nmsth2)
      print("nms2 done")
  else:
     print("nms2 skipped")
  print(time.time() - tt)
  detections6=concatdets(detmsc2,numberlevels)

  im3=copy.deepcopy(im)
  im3=normalize(im3, stat3calib_mean, stat3calib_std, ich)		#NOTE: Typo? Should this be stat3calib or stat3net?
  detections7=applyNet(net3,net3_input,im3,detections6,net3_fov,threshold3,ich)
  print('net3 done')
  print(time.time() - tt)


  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections7
  #print(len(detections))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)
  
  if (len(detections7))==0:
    detections7=[]
    detections8=[]
    detections9=[]
    print(len(detections1),len(detections2),len(detections3),len(detections6),len(detections7),len(detections8),len(detections9))
    return detections6
  detections8,crap = nms(detections7,gnmsth)
  print('gnms done')
  print(time.time() - tt)


  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections8
  #print(len(detections))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)
   
  im3c=copy.deepcopy(im)
  im3c=normalize(im3c, stat3calib_mean, stat3calib_std, ich)
  detections9=applyCalibNet(calib3,calib3_input,im3c,detections8,calib3_fov,threshold3Calib,ich)
  print('calib3 done')
  print('Time for complete cascade:')
  print(time.time() - tt)
  print(len(detections1),len(detections2),len(detections3),len(detections6),len(detections7),len(detections8),len(detections9))
  #print(len(detections1),len(detections2),len(detections3),len(detections4),len(detections5),len(detections6),len(detections7),len(detections8),len(detections9))
  #print(detections9.shape)

  #NOTE: For Debugging:
  #frame_cv = cv2.imread(image)
  #detections = detections9
  #print(len(detections))
  #for j in range(len(detections)):
    #cv2.rectangle(frame_cv, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  #cv2.imshow("Results", frame_cv)
  #cv2.waitKey(0)

  return detections9
  #detections9=verticalExp(detections9)

#Create list of all the images in the images folder.
t = [(image_path + f) for f in listdir(image_path) if f.endswith('.jpg')]

#Torch->TF produces .py files. In this .py files, it is necessary to change the "caffe." to the model name written below (e.g. "app_feat." or "model_12cnet.") for the code to load the net values correctly.
#Additionally, the first model is seperated in two different .py files (app_feat and classif below) by removing the last 3 layers of "app_feat" and putting them as the "classif" layer.
#Only the first net needs to use "extratstide".
app_feat_input = tf.placeholder(tf.float32, shape = (None, None, None, ich))
app_feat = TFNets.app_feat({'input': app_feat_input}, extrastride=extrastride)
app_feat.load(net1_data_path, sess, modelname='app_feat.', ignore_missing=True)
  
classif_input = tf.placeholder(tf.float32, shape = (None, None, None, 32))	#NOTE: Change the 32 if the number of outputs to the last convolution of app_feat changes
classif = TFNets.classif({'input': classif_input})
classif.load(net1_data_path, sess, modelname='classif.', ignore_missing=True)

calib1_input = tf.placeholder(tf.float32, shape = (None, calib1_fov[0], calib1_fov[1], ich))
calib1 = TFNets.model_12cnet({'input': calib1_input})
calib1.load(calib1_data_path, sess, modelname='model_12cnet.')

#TODO:change notation net1->net1 calib1->calib1 etc...
#net2_input = tf.placeholder(tf.float32, shape = (None, net2_fov[0], net2_fov[1], ich))
#net2 = TFNets.model_20net({'input': net2_input})
#net2.load(net2_data_path, sess, modelname='model_20net.')

#calib2_input = tf.placeholder(tf.float32, shape = (None, calib2_fov[0], calib2_fov[1], ich))
#calib2 = TFNets.model_12cnet({'input': calib2_input})
#calib2.load(calib2_data_path, sess, modelname='model_12cnet.')

net3_input = tf.placeholder(tf.float32, shape = (None, net3_fov[0], net3_fov[1], ich))
net3 = TFNets.model_48net({'input': net3_input})
net3.load(net3_data_path, sess, modelname='model_48net.')

calib3_input = tf.placeholder(tf.float32, shape = (None, calib3_fov[0], calib3_fov[1], ich))
calib3 = TFNets.model_48cnet({'input': calib3_input})
calib3.load(calib3_data_path, sess, modelname='model_48cnet.')

s = [0.83, 0.91, 1.0, 1.10, 1.21]
xy = [-0.17, 0, 0.17]
T = np.zeros((45,3))
cnt = 0
for i in range(5):
    for j in range(3):
        for k in range(3):
            cnt=cnt+1
            T[cnt-1][0] = s[i]
            T[cnt-1][1] = xy[j]
            T[cnt-1][2] = xy[k]

for i in range(len(t)):
  image = t[i]
  detections = process()
  frame = cv2.imread(image)
  print(len(detections))
  #print(detections)
  for j in range(len(detections)):
    cv2.rectangle(frame, (int(detections[j][0]), int(detections[j][1])), (int(detections[j][2]), int(detections[j][3])), (0, 0, 255), 2)
  cv2.imshow("Results", frame)
  #Following two lines are only used to save the images with bounding box to file.
  #result_file = result_path + 'Result' + str(i) + '.png'
  #cv2.imwrite(result_file, frame)
  cv2.waitKey(0)
