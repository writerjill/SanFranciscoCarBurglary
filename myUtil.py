import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import pydotplus
import collections

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from shapely import geometry
from shapely.geometry import Point, Polygon

# plot SF basemap with data
def plotBasemap(namePlot ='Break-in density',colorCode = None, df = None, lastPic = False, saveFig = False):

	# display background map 
	img=mpimg.imread('SFlayoutFull.png')
	imgSize = img.shape

	# display boundary
	outline = pd.read_csv('SFB1_UTM.csv')
	outlineX = (outline.Easting - 542094.71)/15.1
	outlineY = (outline.Northing - 4185498.05)/-15.07

	# load points inside boundary
	with open('insidePointsX.txt', 'r') as filehandle:
		insidePointsX = json.load(filehandle)
	with open('insidePointsY.txt', 'r') as filehandle:
		insidePointsY = json.load(filehandle)

	insidePointsXM = (np.asarray(insidePointsX) - 542094.71)/15.1
	insidePointsYM = (np.asarray(insidePointsY) - 4185498.05)/-15.07

	# load density
	with open('density100.txt', 'r') as filehandle:
		density = json.load(filehandle)

	cmapDensity = mpl.colors.ListedColormap(['black','purple','navy','steelblue','darkgreen','yellow'])
	boundsCmap = [0, 5, 20, 50, 100, 300, 600]
	norm = mpl.colors.BoundaryNorm(boundsCmap, cmapDensity.N)		

	# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 7.5), gridspec_kw = {'width_ratios':[3, 0.1]})
	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5.5, 5), gridspec_kw = {'width_ratios':[3, 0.1]})
	ax1.imshow(img)

	if colorCode is None:
		densityColor = density
	else:
		densityColor = colorCode

	ax1.scatter(insidePointsXM, insidePointsYM, s=1, c=densityColor, alpha=0.03, cmap=cmapDensity, norm = norm)
	ax1.plot(outlineX, outlineY, c="black")
	if df is not None:
		ax1.plot((df.Easting - 542094.71)/15.1, (df.Northing - 4185498.05)/-15.07,'r.', ms=1)
	ax1.set_title(namePlot)
	ax1.set_xlim(0, 1000)
	ax1.set_ylim(800, 0)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.get_xaxis().set_ticks([])
	ax1.get_yaxis().set_ticks([])
	mpl.colorbar.ColorbarBase(ax2, cmap=cmapDensity,
	                                alpha = 0.5,
	                                norm = norm,
	                                boundaries=boundsCmap,
	                                extend='neither',
	                                ticks=boundsCmap,
	                                spacing='proportional') 
	ax2.set_ylabel('Number of incidents since 2018', size=12)
	if saveFig:
		plt.savefig(namePlot+'.png')
	plt.show(block=lastPic)

# convert break-in density from raw scale to ordinal classes
def convertRaw2Catagory(raw):
	catagory = raw.copy()
	catagory[raw  < 5] = 1
	catagory[(raw  >= 5) & (raw  < 20)] = 2
	catagory[(raw  >= 20) & (raw  < 50)] = 3
	catagory[(raw  >= 50) & (raw  < 100)] = 4
	catagory[(raw  >= 100) & (raw  < 300)] = 5
	catagory[raw  >= 300] = 6
	return catagory

# convert break-in density from ordinal classes to raw scale 
def convertCatagory2Raw(catagory):
	raw = catagory.copy()
	raw[catagory >= 5.5] = 500
	raw[(catagory >= 4.5) & (catagory < 5.5)] = 200
	raw[(catagory >= 3.5) & (catagory < 4.5)] = 70
	raw[(catagory >= 2.5) & (catagory < 3.5)] = 30
	raw[(catagory >= 1.5) & (catagory < 2.5)] = 10
	raw[catagory < 1.5] = 2
	return raw

# plot tree 
def plotTree(treeName,tree,featureNames):
	treePic_dot = StringIO()
	export_graphviz(tree, out_file=treePic_dot, feature_names=featureNames, filled=True, rounded=True)
	graph = pydotplus.graph_from_dot_data(treePic_dot.getvalue())  
	Image(graph.create_png())
	graph.write_png(treeName+'.png')

# plot feature importances
def plotImp(treeName,tree,featureList,saveFig=False):
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.barh(np.arange(len(featureList)),tree.feature_importances_,align='center')
	ax.set_yticks(np.arange(len(featureList)))
	ax.set_yticklabels(featureList)
	ax.invert_yaxis()  
	ax.set_title(treeName)
	if saveFig:
		plt.savefig(treeName+'(Imp).png')
	plt.show(block=False)

# plot feature importances in descending order
def plotSortedImp(treeName,tree,featureList,numTop=15,saveFig=False):

	print(type(tree.feature_importances_))

	sortedImp = np.sort(tree.feature_importances_)[::-1]
	sortedFeatureList = [x for _,x in sorted(zip(tree.feature_importances_,featureList))]
	sortedFeatureList.reverse()

	fig, ax = plt.subplots(figsize=(7, 5))
	ax.barh(np.arange(numTop), sortedImp[0:numTop], align='center')
	ax.set_yticks(np.arange(numTop))
	ax.set_yticklabels(sortedFeatureList[0:numTop])
	ax.invert_yaxis()  
	ax.set_title(treeName)
	if saveFig:
		plt.savefig(treeName+'(sortedImp).png')
	plt.show()

# get list of indexs of training examples in each grid
def getListofCVindex(dfData,binSize):
	# display boundary
	outline = pd.read_csv('SFB1_UTM.csv')
	boundaryPointList = []
	for X,Y in zip(outline.Easting,outline.Northing):
		boundaryPointList.append(Point(X,Y))
	boundaryPolygon = geometry.Polygon([[p.x, p.y] for p in boundaryPointList])

	X_C = np.arange(binSize/2,1000,binSize)
	Y_C = np.arange(binSize/2,800,binSize)
	Xgrid_C, Ygrid_C = np.meshgrid(X_C,Y_C)

	X_P = np.arange(binSize,1000,binSize)
	Y_P = np.arange(binSize,800,binSize)
	Xgrid_P, Ygrid_P = np.meshgrid(X_P,Y_P)

	Xgrid_C = 15.1*Xgrid_C + 542094.71
	Xgrid_P = 15.1*Xgrid_P + 542094.71

	Ygrid_C = -15.07*Ygrid_C + 4185498.05
	Ygrid_P = -15.07*Ygrid_P + 4185498.05

	# check if corners are in polygon

	isIn = np.zeros((Xgrid_C.shape[0], Xgrid_C.shape[1]), dtype=bool)

	for i in range(Xgrid_P.shape[0]):
		for j in range(Xgrid_P.shape[1]):
			pointsTemp = Point(Xgrid_P[i,j], Ygrid_P[i,j])
			if pointsTemp.within(boundaryPolygon):
				isIn[i,j] = True
				isIn[i+1,j] = True
				isIn[i,j+1] = True
				isIn[i+1,j+1] = True

	XC_in = []
	YC_in = []

	for i in range(isIn.shape[0]):
		for j in range(isIn.shape[1]):
			if isIn[i,j]:
				XC_in.append(Xgrid_C[i,j])
				YC_in.append(Ygrid_C[i,j])

	# bin the index of training data points
	listBinID = []

	for binX,binY in zip(XC_in,YC_in):
		temp = dfData.ID[(np.asarray(dfData.X)>=binX-15.1*binSize/2) & (np.asarray(dfData.X)<=binX+15.1*binSize/2) & (np.asarray(dfData.Y)>=binY-15.07*binSize/2) & (np.asarray(dfData.Y)<=binY+15.07*binSize/2)]
		listBinID.append(temp)

	return listBinID

def convertsMapPoints2UTM_X(x):
	return 15.1*x + 542094.71

def convertsMapPoints2UTM_Y(y)
	return -15.07*x + 4185498.05

def convertsUTM2MapPoints_X(x)
	return (x - 542094.71)/15.1

def convertsUTM2MapPoints_Y(y)
	return (y - 4185498.05)/-15.07

def main():
	plotBasemap(colorCode = None, df = None, lastPic = True, saveFig = True)

if __name__ == '__main__':
	main()