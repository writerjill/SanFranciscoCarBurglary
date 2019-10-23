import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
from shapely import geometry
from shapely.geometry import Point, Polygon
from myUtil import convertsMapPoints2UTM_X, convertsMapPoints2UTM_Y, convertsUTM2MapPoints_X, convertsUTM2MapPoints_Y

# generate random X and Y inside the boundary polygon of SF---------------------------------------------------------------

# load map picture of SF
img=mpimg.imread('SFlayoutFull.png')
imgSize = img.shape

# load boundary polygon of SF
outline = pd.read_csv('SFB1_UTM.csv')

# build boundary object of polygon using shapely
boundaryPointList = []
for X,Y in zip(outline.Easting,outline.Northing):
	boundaryPointList.append(Point(X,Y))
boundaryPolygon = geometry.Polygon([[p.x, p.y] for p in boundaryPointList])

# generate random data points inside the map picture and converts them to be UTM coordinates 
randomPoints = np.random.rand(200000,2)
pointsX = convertsMapPoints2UTM_X(randomPoints[:,0]*imgSize[1])
pointsY = convertsMapPoints2UTM_Y(randomPoints[:,1]*imgSize[0])

# select only the points inside the boundary
for X, Y in zip(pointsX,pointsY):
	pointsTemp = Point(X, Y)
	if pointsTemp.within(boundaryPolygon):
		insidePoints.append(pointsTemp)	

insidePointsX = []
insidePointsY = []
for p in insidePoints:
	insidePointsX.append(p.x)
	insidePointsY.append(p.y)

# generate training dataframe-------------------------------------------------------------------------------------------
nTraining = 100000
dfTraining = pd.DataFrame({'ID': np.arange(0,nTraining,1), 'X': np.asarray(insidePointsX[0:nTraining]), 'Y': np.asarray(insidePointsY[0:nTraining])})

# measure number of attributes
AttractionRadiusNear = 300
AttractionRadiusFar = 600
PoliceRadius = 100
OtherRadius = 100
OtherRadiusFar = 300
TrainingFile = 'dfTraining.csv'

# load the data of interested places
dfReport = pd.read_csv('Report.csv')
dfTopAttractions_UTM = pd.read_csv('dfTopAttractions.csv')
dfPolice_UTM = pd.read_csv('dfPolice.csv')
dfGrocery_UTM = pd.read_csv('dfGrocery.csv')
dfRestaurant_UTM = pd.read_csv('dfRestaurant.csv')
dfHotel_UTM = pd.read_csv('dfHotel.csv')
dfParking_UTM = pd.read_csv('dfParking.csv')

# append distance square (d2) to df
dfRestaurant_UTM = dfRestaurant_UTM.assign(d2 = np.zeros(dfRestaurant_UTM.shape[0]))
dfHotel_UTM = dfHotel_UTM.assign(d2 = np.zeros(dfHotel_UTM.shape[0]))
dfParking_UTM = dfParking_UTM.assign(d2 = np.zeros(dfParking_UTM.shape[0]))

# initialize feature lists
numReport = []
numTopAttractionsNear = []
numTopAttractionsFar = []
numPolice = []
numPoliceFar = []
numGrocery = []
numGroceryFar = []
numRestaurant = []
numRestaurantFar = []
numRestaurantPrice0 = []
numRestaurantPrice0Far = []
numRestaurantPrice1 = []
numRestaurantPrice1Far = []
numRestaurantPrice2 = []
numRestaurantPrice2Far = []
numRestaurantPrice34 = []
numRestaurantPrice34Far = []
numRestaurantNumReviewL333 = []
numRestaurantNumReviewL333Far = []
numRestaurantNumReviewF333T666 = []
numRestaurantNumReviewF333T666Far = []
numRestaurantNumReviewF666T1000 = []
numRestaurantNumReviewF666T1000Far = []
numRestaurantNumReviewG1000 = []
numRestaurantNumReviewG1000Far = []
numHotel = []
numHotelFar = []
numHotelNumReviewL500 = []
numHotelNumReviewL500Far = []
numHotelNumReviewF500T1000 = []
numHotelNumReviewF500T1000Far = []
numHotelNumReviewG1000 = []
numHotelNumReviewG1000Far = []
numParking = []
numParkingFar = []
numParkingReviewL3 = []
numParkingReviewL3Far = []
numParkingReviewF3T4 = []
numParkingReviewF3T4Far = []
numParkingReviewG4 = []
numParkingReviewG4Far = []
numParkingReviewNumL100 = []
numParkingReviewNumL100Far = []
numParkingReviewNumG100 = []
numParkingReviewNumG100Far = []


# building the training data frame
for X,Y in zip(dfTraining.X, dfTraining.Y):
	# calculate distance square
	d2Report = (dfReport.Easting - X)**2 + (dfReport.Northing - Y)**2
	d2TopAttractions = (dfTopAttractions_UTM.Easting - X)**2 + (dfTopAttractions_UTM.Northing - Y)**2
	d2Police = (dfPolice_UTM.Easting - X)**2 + (dfPolice_UTM.Northing - Y)**2
	d2Grocery = (dfGrocery_UTM.Easting - X)**2 + (dfGrocery_UTM.Northing - Y)**2
	d2Restaurant = (dfRestaurant_UTM.Easting - X)**2 + (dfRestaurant_UTM.Northing - Y)**2	
	d2Hotel = (dfHotel_UTM.Easting - X)**2 + (dfHotel_UTM.Northing - Y)**2
	d2Parking = (dfParking_UTM.Easting - X)**2 + (dfParking_UTM.Northing - Y)**2

	dfRestaurant_UTM.d2 = d2Restaurant
	dfHotel_UTM.d2 = d2Hotel
	dfParking_UTM.d2 = d2Parking	

	numReport.append(len(d2Report[d2Report <= OtherRadius**2]))
	numTopAttractionsNear.append(len(d2TopAttractions[d2TopAttractions <= AttractionRadiusNear**2]))
	numTopAttractionsFar.append(len(d2TopAttractions[d2TopAttractions <= AttractionRadiusFar**2]))
	numPolice.append(len(d2Police[d2Police <= PoliceRadius**2]))
	numGrocery.append(len(d2Grocery[d2Grocery <= OtherRadius**2]))
	numRestaurant.append(len(d2Restaurant[d2Restaurant <= OtherRadius**2]))
	numRestaurantPrice0.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Price']) < 1)].shape[0])
	numRestaurantPrice1.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 1) & (np.asarray(dfRestaurant_UTM['Price']) < 2)].shape[0])
	numRestaurantPrice2.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 2) & (np.asarray(dfRestaurant_UTM['Price']) < 3)].shape[0])
	numRestaurantPrice34.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 3)].shape[0])
	numRestaurantNumReviewL333.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 333)].shape[0])
	numRestaurantNumReviewF333T666.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 333) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 666)].shape[0])
	numRestaurantNumReviewF666T1000.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 666) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 1000)].shape[0])
	numRestaurantNumReviewG1000.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 1000)].shape[0])
	numHotel.append(len(d2Hotel[d2Hotel <= OtherRadius**2]))
	numHotelNumReviewL500.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) < 500)].shape[0])
	numHotelNumReviewF500T1000.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) >= 500) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) < 1000)].shape[0])
	numHotelNumReviewG1000.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) >= 1000)].shape[0])
	numParking.append(len(d2Parking[d2Parking <= OtherRadius**2]))
	numParkingReviewL3.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfParking_UTM['Review']) < 3)].shape[0])
	numParkingReviewF3T4.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfParking_UTM['Review']) >= 3) & (np.asarray(dfParking_UTM['Review']) < 4)].shape[0])
	numParkingReviewG4.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfParking_UTM['Review']) >= 4)].shape[0])
	numParkingReviewNumL100.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfParking_UTM['Number_Of_Reviewers']) < 100)].shape[0])
	numParkingReviewNumG100.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadius**2) & (np.asarray(dfParking_UTM['Number_Of_Reviewers']) >= 100)].shape[0])

	numPoliceFar.append(len(d2Police[d2Police <= OtherRadiusFar**2]))
	numGroceryFar.append(len(d2Grocery[d2Grocery <= OtherRadiusFar**2]))
	numRestaurantFar.append(len(d2Restaurant[d2Restaurant <= OtherRadius**2]))
	numRestaurantPrice0Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Price']) < 1)].shape[0])
	numRestaurantPrice1Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 1) & (np.asarray(dfRestaurant_UTM['Price']) < 2)].shape[0])
	numRestaurantPrice2Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 2) & (np.asarray(dfRestaurant_UTM['Price']) < 3)].shape[0])
	numRestaurantPrice34Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Price']) >= 3)].shape[0])
	numRestaurantNumReviewL333Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 333)].shape[0])
	numRestaurantNumReviewF333T666Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 333) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 666)].shape[0])
	numRestaurantNumReviewF666T1000Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 666) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) < 1000)].shape[0])
	numRestaurantNumReviewG1000Far.append(dfRestaurant_UTM.loc[(np.asarray(dfRestaurant_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfRestaurant_UTM['Number_Of_Reviewers']) >= 1000)].shape[0])
	numHotelFar.append(len(d2Hotel[d2Hotel <= OtherRadiusFar**2]))
	numHotelNumReviewL500Far.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) < 500)].shape[0])
	numHotelNumReviewF500T1000Far.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) >= 500) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) < 1000)].shape[0])
	numHotelNumReviewG1000Far.append(dfHotel_UTM.loc[(np.asarray(dfHotel_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfHotel_UTM['Number_Of_Reviewers']) >= 1000)].shape[0])
	numParkingFar.append(len(d2Parking[d2Parking <= OtherRadiusFar**2]))
	numParkingReviewL3Far.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfParking_UTM['Review']) < 3)].shape[0])
	numParkingReviewF3T4Far.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfParking_UTM['Review']) >= 3) & (np.asarray(dfParking_UTM['Review']) < 4)].shape[0])
	numParkingReviewG4Far.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfParking_UTM['Review']) >= 4)].shape[0])
	numParkingReviewNumL100Far.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfParking_UTM['Number_Of_Reviewers']) < 100)].shape[0])
	numParkingReviewNumG100Far.append(dfParking_UTM.loc[(np.asarray(dfParking_UTM['d2']) <= OtherRadiusFar**2) & (np.asarray(dfParking_UTM['Number_Of_Reviewers']) >= 100)].shape[0])

	
dfTraining = pd.merge(dfTraining, pd.DataFrame({'ID': np.arange(0,nTraining,1),
												'numReport': numReport,
												'numTopAttractionsNear': numTopAttractionsNear,
												'numTopAttractionsFar': numTopAttractionsFar,
												'numPolice': numPolice,
												'numGrocery': numGrocery,
												'numRestaurant': numRestaurant,
												'numRestaurantPrice0': numRestaurantPrice0,
												'numRestaurantPrice1': numRestaurantPrice1,
												'numRestaurantPrice2': numRestaurantPrice2,
												'numRestaurantPrice34': numRestaurantPrice34,
												'numRestaurantNumReviewL333': numRestaurantNumReviewL333,
												'numRestaurantNumReviewF333T666': numRestaurantNumReviewF333T666,
												'numRestaurantNumReviewF666T1000': numRestaurantNumReviewF666T1000,
												'numRestaurantNumReviewG1000': numRestaurantNumReviewG1000,
												'numHotel': numHotel,
												'numHotelNumReviewL500': numHotelNumReviewL500,
												'numHotelNumReviewF500T1000': numHotelNumReviewF500T1000,
												'numHotelNumReviewG1000': numHotelNumReviewG1000,
												'numParking': numParking,
												'numParkingReviewL3': numParkingReviewL3,
												'numParkingReviewF3T4': numParkingReviewF3T4,
												'numParkingReviewG4': numParkingReviewG4,
												'numParkingReviewNumL100': numParkingReviewNumL100,
												'numParkingReviewNumG100': numParkingReviewNumG100,
												'numPoliceFar': numPoliceFar,
												'numGroceryFar': numGroceryFar,
												'numRestaurantFar': numRestaurantFar,
												'numRestaurantPrice0Far': numRestaurantPrice0Far,
												'numRestaurantPrice1Far': numRestaurantPrice1Far,
												'numRestaurantPrice2Far': numRestaurantPrice2Far,
												'numRestaurantPrice34Far': numRestaurantPrice34Far,
												'numRestaurantNumReviewL333Far': numRestaurantNumReviewL333Far,
												'numRestaurantNumReviewF333T666Far': numRestaurantNumReviewF333T666Far,
												'numRestaurantNumReviewF666T1000Far': numRestaurantNumReviewF666T1000Far,
												'numRestaurantNumReviewG1000Far': numRestaurantNumReviewG1000Far,
												'numHotelFar': numHotelFar,
												'numHotelNumReviewL500Far': numHotelNumReviewL500Far,
												'numHotelNumReviewF500T1000Far': numHotelNumReviewF500T1000Far,
												'numHotelNumReviewG1000Far': numHotelNumReviewG1000Far,
												'numParkingFar': numParkingFar,
												'numParkingReviewL3Far': numParkingReviewL3Far,
												'numParkingReviewF3T4Far': numParkingReviewF3T4Far,
												'numParkingReviewG4Far': numParkingReviewG4Far,
												'numParkingReviewNumL100Far': numParkingReviewNumL100Far,
												'numParkingReviewNumG100Far': numParkingReviewNumG100Far}), on='ID')

dfTraining.to_csv(TrainingFile)