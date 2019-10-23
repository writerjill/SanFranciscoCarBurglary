import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from pyproj import Proj

# This file converts address to UTM coordinates. Then add the Easting and Northing columns to the dataframe

geolocator = Nominatim(user_agent="parkingCoordinate")

inputFile = 'XXX.csv'
outputFile = 'XXX_UTM.csv'

df = pd.read_csv(inputFile)
df['latitude'] = np.nan
df['longitude'] = np.nan

for index, row in df.iterrows():
	# convert address to lat and long
	tempAddress = row['Address'] + ", San Francisco, CA"
	location = geolocator.geocode(tempAddress)
	df.at[index,'latitude'] = location.latitude
	df.at[index,'longitude'] = location.longitude
	# convert lat and long to UTM
	UTMx, UTMy = myProj(row['longitude'], row['latitude'])
	df.at[index,'Easting'] = UTMx
	df.at[index,'Northing'] = UTMy
	# save the outputs every 50 rows in case of connection loss
	if index%50 == 0:
		df.to_csv(outputFile)

df.to_csv(outputFile)