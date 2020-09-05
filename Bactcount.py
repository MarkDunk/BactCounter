import argparse
import numpy as np
import cv2 as cv

def K_Means(img):
	flat = img.flatten()

	flat = np.float32(flat)

	params = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
	K = 2
	
	ret,label,center=cv.kmeans(flat,K,None,params,10,cv.KMEANS_RANDOM_CENTERS)

	center = np.uint8(center)
	res = center[label.flatten()]
	res = res.reshape((img.shape))

	threshold = np.nanmin(center)+1
	ret,binary = cv.threshold(res, threshold, 255, cv.THRESH_BINARY)

	return(binary)

def Blobby(binary, ogimg, params):

	blobber = cv.SimpleBlobDetector_create(params)

	blobs = blobber.detect(binary)

	outline = cv.drawKeypoints(ogimg, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return(blobs, outline)


class BCount:
	def __init__(self, count=0):
		self.count = count
	def Read(self, fn):
		self.img = cv.imread(fn, 0)
		self.img = cv.resize(self.img, (500,500), interpolation=cv.INTER_NEAREST)

	def ReadBlue(self, fn):
		colorimg = cv.imread(fn)
		hsv = cv.cvtColor(colorimg, cv.COLOR_RGB2HSV)
		self.img = hsv[:,:,0]
		self.img = cv.bitwise_not(self.img)

	def Threshold(self):
		mask = K_Means(self.img)

		mask = cv.medianBlur(mask, 3)

		cropped = cv.bitwise_and(self.img, self.img, mask=mask)

		adbin = cv.adaptiveThreshold(cropped,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, -2)

		adbin = cv.medianBlur(adbin, 3)

		kernel = np.ones((3,3), np.uint8)

		self.binary = cv.dilate(adbin, kernel, iterations=1)

	def Detect(self):
		blobs_param = cv.SimpleBlobDetector_Params()

		blobs_param.minThreshold = 200
		blobs_param.maxThreshold = 255

		blobs_param.filterByInertia = False
		blobs_param.minInertiaRatio = .3

		blobs_param.filterByColor = True
		blobs_param.blobColor = 255

		blobs_param.filterByArea = True
		blobs_param.minArea = 9
		blobs_param.maxArea = 75

		blobs_param.minDistBetweenBlobs = 0

		self.count = Blobby(self.binary, self.img, blobs_param)

		return(self.count)


parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Filename of image')
parser.add_argument('type', help='Standard Plate (SPC) or Blue-White (BW)')
args = parser.parse_args()
if args.type == 'BW':
	white = BCount()
	white.Read(args.filename)
	white.Threshold()
	total = white.Detect()

	blue = BCount()
	blue.Read_Blue(args.filename)
	blue.Threshold()
	total += blue.Detect()
	print(total)

elif args.type == 'SPC':
	white = BCount()
	white.Read(args.filename)
	white.Threshold()
	total = white.Detect()
	print(total)
else:
	print('Invalid type selected')
