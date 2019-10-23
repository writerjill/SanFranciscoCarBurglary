import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import pickle
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from myUtil import plotBasemap, convertRaw2Catagory, convertCatagory2Raw, plotTree, plotImp, getListofCVindex

# import dataset-------------------------------------------------------------------------------------------------------------
dfData = pd.read_csv('dfTraining.csv')
featureList = ['numTopAttractionsNear','numTopAttractionsFar',
               'numParking','numParkingReviewL3','numParkingReviewF3T4','numParkingReviewG4','numParkingReviewNumL100','numParkingReviewNumG100',
               'numRestaurant','numRestaurantPrice0','numRestaurantPrice1','numRestaurantPrice2','numRestaurantPrice34',
               'numRestaurantNumReviewL333','numRestaurantNumReviewF333T666','numRestaurantNumReviewF666T1000','numRestaurantNumReviewG1000',
               'numGrocery','numPolice','numHotel', 'numHotelNumReviewL500','numHotelNumReviewF500T1000','numHotelNumReviewG1000',
               'numParkingFar','numParkingReviewL3Far','numParkingReviewF3T4Far','numParkingReviewG4Far','numParkingReviewNumL100Far','numParkingReviewNumG100Far',
               'numRestaurantFar','numRestaurantPrice0Far','numRestaurantPrice1Far','numRestaurantPrice2Far','numRestaurantPrice34Far',
               'numRestaurantNumReviewL333Far','numRestaurantNumReviewF333T666Far','numRestaurantNumReviewF666T1000Far','numRestaurantNumReviewG1000Far',
               'numGroceryFar','numPoliceFar','numHotelFar', 'numHotelNumReviewL500Far','numHotelNumReviewF500T1000Far','numHotelNumReviewG1000Far']
dataX = dfData[featureList]
dataYRaw = dfData['numReport'].as_matrix()

# convert raw values to ordinal classes (1,2,3,4,5,and 6)
dataY = convertRaw2Catagory(dataYRaw)

# load grid inside SF for ploting purpose
dfDataMap = pd.read_csv('dfMap.csv')
dataMapX = dfDataMap[featureList]
dataMapY = dfDataMap['numReport'].as_matrix()
dataMapYRound = convertRaw2Catagory(dataMapY)

# perform linear regression as a baseline-------------------------------------------------------------------------------------

# train the linear model
LinearRegressor = LinearRegression()  
LinearRegressor.fit(dataX, dataY) 

# plot over SF map and calculate accuracy
y_linear = LinearRegressor.predict(dataMapX)
y_linearRaw = convertCatagory2Raw(y_linear)

y_linearRound = y_linear.copy()
y_linearRound = np.rint(np.asarray(y_linearRound))
y_linearRound[y_linearRound < 1] = 1
y_linearRound[y_linearRound > 6] = 6

print('Confusion matrix of y_linear')
print(confusion_matrix(dataMapYRound,y_linearRound))
print('Accuracy of y_linear is %1.4f' % (accuracy_score(dataMapYRound,y_linearRound)))

# setup indexs for cross validation-------------------------------------------------------------------------------------------
listBinID = getListofCVindex(dfData,binSize=50)
random.shuffle(listBinID)
numCV = 5

# Train tree model------------------------------------------------------------------------------------------------------------

# CV for tree
rangeMaxDepth = np.arange(1,41,1)
TreeTrainError = np.zeros((len(rangeMaxDepth),numCV))
TreeTestError = np.zeros((len(rangeMaxDepth),numCV))

for i, maxDepth in enumerate(rangeMaxDepth):
	for cv in range(numCV):
		treeTemp = DecisionTreeRegressor(random_state = 0, max_depth = maxDepth)  
		listBinIDTemp = listBinID.copy()
		testIndex = [int(cv*numlistBinID/numCV):int((cv+1)*numlistBinID/numCV)]
		del listBinIDTemp[testIndex]
		treeTemp.fit(X=dataX.iloc[list(np.concatenate(listBinIDTemp)),:], y=dataY[np.concatenate(listBinIDTemp)])
		TreeTrainError[i,cv] = mean_squared_error(dataY[np.concatenate(listBinIDTemp)], treeTemp.predict(dataX.iloc[list(np.concatenate(listBinIDTemp)),:]))
		TreeTestError[i,cv] = mean_squared_error(dataY[testIndex], treeTemp.predict(dataX.iloc[testIndex,:]))

meanTreeTrainError = np.mean(TreeTrainError, axis = 1)
meanTreeTestError = np.mean(TreeTestError, axis = 1)

ChosenMaxDepth = rangeMaxDepth[np.argmin(meanTreeTestError)]
print('ChosenMaxDepth for tree is '+str(ChosenMaxDepth))

treeRegressor = DecisionTreeRegressor(random_state = 0, max_depth = ChosenMaxDepth)
treeRegressor.fit(X=dataX,y=dataY)

# Dump the trained decision tree classifier with Pickle
decision_tree_model_pkl = open('treeRegressor.pkl', 'wb')
pickle.dump(treeRegressor, decision_tree_model_pkl)
decision_tree_model_pkl.close()

# plot features importance
plotImp(treeName = 'TreeRegressor',tree=treeRegressor,featureList=featureList,saveFig=True)

# predict over map
y_tree = treeRegressor.predict(dataMapX)
y_treeRaw = convertCatagory2Raw(y_tree)

y_treeRound = y_tree.copy()
y_treeRound = np.rint(np.asarray(y_treeRound))
y_treeRound[y_treeRound < 1] = 1
y_treeRound[y_treeRound > 6] = 6

print('Confusion matrix of y_tree8')
print(confusion_matrix(dataMapYRound,y_treeRound))
print('Accuracy of y_tree is %1.4f' % (accuracy_score(dataMapYRound,y_treeRound)))

plotBasemap(namePlot ='Tree)', colorCode = y_treeRaw, df = None, lastPic = False, saveFig = True)

# Train Random forest model------------------------------------------------------------------------------------------------------------

RFTrainError = np.zeros((len(rangeMaxDepth),numCV))
RFTestError = np.zeros((len(rangeMaxDepth),numCV))

for i, maxDepth in enumerate(rangeMaxDepth):
	for cv in range(numCV):
		RFTemp = RandomForestRegressor(random_state = 0, n_estimators = 50, max_features = "sqrt", max_depth = maxDepth)  
		listBinIDTemp = listBinID.copy()
		testIndex = [int(cv*numlistBinID/numCV):int((cv+1)*numlistBinID/numCV)]
		del listBinIDTemp[testIndex]
		RFTemp.fit(X=dataX.iloc[list(np.concatenate(listBinIDTemp)),:], y=dataY[np.concatenate(listBinIDTemp)])
		RFTrainError[i,cv] = mean_squared_error(dataY[np.concatenate(listBinIDTemp)], RFTemp.predict(dataX.iloc[list(np.concatenate(listBinIDTemp)),:]))
		RFTestError[i,cv] = mean_squared_error(dataY[testIndex], RFTemp.predict(dataX.iloc[testIndex,:]))

meanRFTrainError = np.mean(RFTrainError, axis = 1)
meanRFTestError = np.mean(RFTestError, axis = 1)

ChosenMaxDepth = rangeMaxDepth[np.argmin(meanRFTestError)]
print('ChosenMaxDepth of y_RF is '+str(ChosenMaxDepth))

RFRegressor = RandomForestRegressor(random_state = 0, n_estimators = 50, max_features = "sqrt", max_depth = ChosenMaxDepth)  
RFRegressor.fit(X=dataX,y=dataY)

# Dump the trained decision tree classifier with Pickle
decision_tree_model_pkl = open('RFRegressor.pkl', 'wb')
pickle.dump(RFRegressor, decision_tree_model_pkl)
decision_tree_model_pkl.close()

# plot features importance
plotImp(treeName = 'RFRegressor',tree=RFRegressor,featureList=featureList,saveFig=True)

# predict over map
y_RF = RFRegressor.predict(dataMapX)
y_RFRaw = convertCatagory2Raw(y_RF)

y_RFRound = y_RF.copy()
y_RFRound = np.rint(np.asarray(y_RFRound))
y_RFRound[y_RFRound < 1] = 1
y_RFRound[y_RFRound > 6] = 6

print('Confusion matrix of y_RF')
print(confusion_matrix(dataMapYRound,y_RFRound))
print('Accuracy of y_RF is %1.4f' % (accuracy_score(dataMapYRound,y_RFRound)))

plotBasemap(namePlot ='RF', colorCode = y_RFRaw, df = None, lastPic = False, saveFig = True)

# Train GBM model------------------------------------------------------------------------------------------------------------

# CV for numTree and learning rate: try different learning and test which one give the best result with n_estimators around 80
rangeTree = np.arange(20,101,20)

GBMTrainError_alpha = np.zeros((len(rangeTree ),numCV))
GBMTestError_alpha = np.zeros((len(rangeTree ),numCV))

for i, numTree in enumerate(rangeTree):
	for cv in range(numCV):
		GBMTemp = GradientBoostingRegressor(learning_rate=1, random_state=0, max_features='sqrt', subsample=0.8, n_estimators = numTree)
		listBinIDTemp = listBinID.copy()
		testIndex = [int(cv*numlistBinID/numCV):int((cv+1)*numlistBinID/numCV)]
		del listBinIDTemp[testIndex]
		GBMTemp.fit(X=dataX.iloc[list(np.concatenate(listBinIDTemp)),:], y=dataY[np.concatenate(listBinIDTemp)])
		GBMTrainError_alpha[i,cv] = mean_squared_error(dataY[np.concatenate(listBinIDTemp)], GBMTemp.predict(dataX.iloc[list(np.concatenate(listBinIDTemp)),:]))
		GBMTestError_alpha[i,cv] = mean_squared_error(dataY[testIndex], GBMTemp.predict(dataX.iloc[testIndex,:]))

meanGBMTrainError_alpha = np.mean(GBMTrainError_alpha, axis = 1)
meanGBMTestError_alpha = np.mean(GBMTestError_alpha axis = 1)

plt.plot(rangeTree ,meanGBMTrainError_alpha,'r')
plt.plot(rangeTree ,meanGBMTestError_alpha,'b')
plt.show()

# CV for max depth
rangeMaxDepth = np.arange(1,10,1)

GBMTrainError = np.zeros((len(rangeMaxDepth),numCV))
GBMTestError = np.zeros((len(rangeMaxDepth),numCV))

for i, maxDepth in enumerate(rangeMaxDepth):
	for cv in range(numCV):
		regressorOrdCatTemp = GradientBoostingRegressor(learning_rate=1, random_state=0, max_features='sqrt', subsample=0.8, n_estimators = 70, max_depth = maxDepth)
		listBinIDTemp = listBinID.copy()
		testIndex = [int(cv*numlistBinID/numCV):int((cv+1)*numlistBinID/numCV)]
		del listBinIDTemp[testIndex]
		regressorOrdCatTemp.fit(X=dataX.iloc[list(np.concatenate(listBinIDTemp)),:], y=dataY[np.concatenate(listBinIDTemp)])
		GBMTrainError[i,cv] = mean_squared_error(dataY[np.concatenate(listBinIDTemp)], regressorOrdCatTemp.predict(dataX.iloc[list(np.concatenate(listBinIDTemp)),:]))
	    GBMTestError[i,cv] = mean_squared_error(dataY[testIndex], regressorOrdCatTemp.predict(dataX.iloc[testIndex,:]))

meanGBMTrainError = np.mean(GBMTrainError, axis = 1)
meanGBMTestError = np.mean(GBMTestError, axis = 1)

ChosenMaxDepth = rangeMaxDepth[np.argmin(meanGBMTestError)]
print('ChosenMaxDepth of y_GBM is '+str(ChosenMaxDepth))

# fit the model
GBMRegressor = GradientBoostingRegressor(learning_rate=1, random_state=0, max_features='sqrt', subsample=0.8, n_estimators = 70, max_depth = ChosenMaxDepth)
GBMRegressor.fit(X=dataX,y=dataY)

# Dump the trained decision tree classifier with Pickle
decision_tree_model_pkl = open('GBMRegressor.pkl', 'wb')
pickle.dump(GBMRegressor, decision_tree_model_pkl)
decision_tree_model_pkl.close()

# plot features importance
plotImp(treeName = 'GBMRegressor',tree=GBMRegressor,featureList=featureList,saveFig=True)

# predict over map
y_GBM = GBMRegressor.predict(dataMapX)
y_GBMRaw = convertCatagory2Raw(y_GBM)

y_GBMRound = y_GBM.copy()
y_GBMRound = np.rint(np.asarray(y_GBMRound))
y_GBMRound[y_GBMRound < 1] = 1
y_GBMRound[y_GBMRound > 6] = 6

print('Confusion matrix of y_GBM')
print(confusion_matrix(dataMapYRound,y_GBMRound))
print('Accuracy of y_GBM is %1.4f' % (accuracy_score(dataMapYRound,y_GBMRound)))

plotBasemap(namePlot ='GBM_maxDepth_CV (ordinal catagory)', colorCode = y_GBMRaw, df = None, lastPic = True, saveFig = True)