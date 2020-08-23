# -*- coding: utf-8 -*-
"""

@author: THO84231
"""

#libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
#API libraries
from urllib.parse import urlencode
import requests
import json
import time 
 
#import london boroughs shapefile

#this reads in of boroughs data file downloaded from london data store.
boroughs = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp')
#the initial file is in british national grid. 
boroughs.crs = {'init' :'epsg:27700'}
boroughs.crs
#convert british national grid to wgs84
boroughs = boroughs.to_crs("EPSG:4326")
boroughs.crs

#count how many boroughs in London
boroughs.count
#check geom column. 
boroughs['geometry']

#check plot
boroughs.plot()

#calculate centroid of the boroughs
boroughs['centroid'] = boroughs.centroid
boroughs.head()
boroughs.columns

#create a new gdf of just the centroids. 
borough_centroids = boroughs.copy()
#reset the geometry column of this new gdf to equal the centroid column in order to plot the points. 
borough_centroids['geometry'] = borough_centroids['centroid']

borough_centroids.crs

#plot centroids on map
fig, ax = plt.subplots(figsize=(10,10))
borough_centroids.plot(ax=ax, alpha=0.4)
borough_centroids.plot(ax=ax, markersize=10, color='red', marker ='o')

#check buffer distance to cover all of london. Cannot just do 6000m while in WGS84 crs need to transform using pyproj. However,  
#this was done in Arc as Pyproj was failing to work at the time of coding. 6000m buffer from all centroids covers all of london. 

#examine the centroid column. 
boroughs['centroid'] 
#when inputting these coordinates in this order to google maps the location is misplaced

#a new version of the boroughs and centroids must be created however, due to the order that Google reads in as Lat Lon 
boroughs_googleGEOM =boroughs.copy()

#remap the order of the Lat Lon for Google to understand correct location
boroughs_googleGEOM['geometry'] = boroughs.geometry.map(lambda polygon:shapely.ops.transform(lambda x, y: (y, x), polygon))

#check orig vs new geom 
boroughs['geometry'].head()
boroughs_googleGEOM['geometry'].head()


#create copy of corrected data ready to calculate centroids based on the newly ordered geom. 
boroughcentroids = boroughs_googleGEOM.copy()
#create new column and calculate centroid of the polygons. 
boroughcentroids['centroid_geom_google'] = boroughs_googleGEOM.centroid
boroughcentroids.drop(['centroid'], axis=1, inplace=True)

#check borough centroids is correct with geom for boroughs in the correct order for google, and correct centroids 
boroughcentroids.columns
boroughcentroids.head()
boroughcentroids[['geometry', 'centroid_geom_google']]
type(boroughcentroids)


#need to extract the lat lon coords from the column into separate columns to then merge as string for use in API call
boroughcentroids['lat'] = boroughcentroids.centroid_geom_google.apply(lambda a: a.x)
boroughcentroids['lon'] = boroughcentroids.centroid_geom_google.apply(lambda a: a.y)
#check this has worked
boroughcentroids[['centroid_geom_google', 'lat', 'lon']]

#merge the resultant columns into one string column for API loop
boroughcentroids['centroidlatlongstring'] = boroughcentroids['lat'].astype(str)+","+boroughcentroids['lon'].astype(str)

#put the column into a new series on its own to parse with the API loop
boroughcentroids_string = boroughcentroids['centroidlatlongstring'] 
type(boroughcentroids_string)
#check
boroughcentroids_string

boroughcentroids_string.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/centroidlatlongstring.csv')


#%% API call 

#define api key

api_key = #blank for this upload into public domain. 

#define the url endpoint
url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


#create empty list for storing results 
mosque_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid as a search point and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written to  a file. 
#3 loops here. first one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_mosq = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "mosque"}
    params_encoded_mosq = urlencode(params_mosq)
    urlfinal_mosq = f"{url}?{params_encoded_mosq}"
    print(urlfinal_mosq)
    r_final_1_mosq = requests.get(urlfinal_mosq)
    r_finaljson_1_mosq = json.loads(r_final_1_mosq.text)  
    mosque_output_results.append(r_finaljson_1_mosq)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_mosq:
        print("yes 2nd page token found")
        #sleep to wait for token to load for next page. 
        time.sleep(3)
        params_2_mosq = {"key": api_key,"pagetoken":r_finaljson_1_mosq['next_page_token']}
        params_encoded_2_mosq = urlencode(params_2_mosq)
        urlfinal2_mosq = f"{url}?{params_encoded_2_mosq}"
        print(urlfinal2_mosq)
        r_final_2_mosq=requests.get(urlfinal2_mosq)
        r_finaljson_2_mosq = r_final_2_mosq.json()
        mosque_output_results.append(r_finaljson_2_mosq)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_mosq:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_mosq = {"key": api_key,"pagetoken":r_finaljson_2_mosq['next_page_token']}
                params_encoded_3_mosq = urlencode(params_3_mosq)
                urlfinal3_mosq = f"{url}?{params_encoded_3_mosq}"
                print(urlfinal3_mosq)
                r_final_3_mosq = requests.get(urlfinal3_mosq)
                r_finaljson_3_mosq = r_final_3_mosq.json()
                mosque_output_results.append(r_finaljson_3_mosq)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
    #save the results to files.           
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/mosque_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(mosque_output_results, outfile, indent=4)
#test
len(mosque_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(mosque_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(mosque_output_results[3]['results'][1]['name'])

#function to get all names,lat,longs from all mosques found in the API results.  
#blank list for results
all_mosques_london = []

def get_latlons (L):
    for everything in mosque_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(all_mosques_london)
len(all_mosques_london)
type(all_mosques_london)
print(all_mosques_london)
#create a data  frame from the list
all_mosques_london_df = pd.DataFrame(all_mosques_london)
all_mosques_london_df.head()


#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
all_mosques_london_df_latlonextracted = pd.concat([all_mosques_london_df.drop([1], axis=1), all_mosques_london_df[1].apply(pd.Series)], axis=1)
all_mosques_london_df_latlonextracted.head()
all_mosques_london_df_latlonextracted[0]

#remove duplicates
all_mosques_london_df_latlonextracted_nodups = all_mosques_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
#save as csv
all_mosques_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/mosques_output_results_DF_latlonextract_nodups.csv")


#read in as csv to make columns string for writnig to shapefile 
all_mosques_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/mosques_output_results_DF_latlonextract_nodups.csv")
all_mosques_london_df_latlonextracted_nodups.head()
all_mosques_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
all_mosques_london_Gdf = gpd.GeoDataFrame(all_mosques_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(all_mosques_london_df_latlonextracted_nodups.lng, all_mosques_london_df_latlonextracted_nodups.lat))
all_mosques_london_Gdf['0']

all_mosques_london_Gdf.crs
#set CRS as WGS84
all_mosques_london_Gdf.crs = {'init' :'epsg:4326'}
all_mosques_london_Gdf.crs

all_mosques_london_Gdf.plot()

all_mosques_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_mosques_london_Gdf.shp')



#%% API CALL FOR CHURCH


#create empty list for storing results 
church_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid as a search point and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written to  a file. 
#3 loops here. first one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_chu = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "church"}
    params_encoded_chu = urlencode(params_chu)
    urlfinal_chu = f"{url}?{params_encoded_chu}"
    print(urlfinal_chu)
    r_final_1_chu = requests.get(urlfinal_chu)
    r_finaljson_1_chu = json.loads(r_final_1_chu.text)  
    church_output_results.append(r_finaljson_1_chu)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_chu:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_chu = {"key": api_key,"pagetoken":r_finaljson_1_chu['next_page_token']}
        params_encoded_2_chu = urlencode(params_2_chu)
        urlfinal2_chu = f"{url}?{params_encoded_2_chu}"
        print(urlfinal2_chu)
        r_final_2_chu=requests.get(urlfinal2_chu)
        r_finaljson_2_chu = r_final_2_chu.json()
        church_output_results.append(r_finaljson_2_chu)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_chu:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_chu = {"key": api_key,"pagetoken":r_finaljson_2_chu['next_page_token']}
                params_encoded_3_chu = urlencode(params_3_chu)
                urlfinal3_chu = f"{url}?{params_encoded_3_chu}"
                print(urlfinal3_chu)
                r_final_3_chu = requests.get(urlfinal3_chu)
                r_finaljson_3_chu = r_final_3_chu.json()
                church_output_results.append(r_finaljson_3_chu)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/church_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(church_output_results, outfile, indent=4)
#test
len(church_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(church_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(church_output_results[3]['results'][1]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
all_churches_london = []

def get_latlons (L):
    for everything in church_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(all_churches_london)

len(all_churches_london)
type(all_churches_london)

print(all_churches_london)

#create a data  frame from the list
all_churches_london_df = pd.DataFrame(all_churches_london)
all_churches_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
all_churches_london_df_latlonextracted = pd.concat([all_churches_london_df.drop([1], axis=1), all_churches_london_df[1].apply(pd.Series)], axis=1)
all_churches_london_df_latlonextracted.head()
#check
all_churches_london_df_latlonextracted[0]

#remove duplicates
all_churches_london_df_latlonextracted_nodups = all_churches_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
all_churches_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/churches_output_results_DF_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
all_churches_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/churches_output_results_DF_latlonextract_nodups.csv")
all_churches_london_df_latlonextracted_nodups.head()
all_churches_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
all_churches_london_Gdf = gpd.GeoDataFrame(all_churches_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(all_churches_london_df_latlonextracted_nodups.lng, all_churches_london_df_latlonextracted_nodups.lat))
all_churches_london_Gdf['0']

all_churches_london_Gdf.crs

all_churches_london_Gdf.crs = {'init' :'epsg:4326'}
all_churches_london_Gdf.crs

all_churches_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_churches_london_Gdf.shp')




#%% API call for synagogues


#create empty list for storing results 
synagogue_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid as a search point and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written to  a file. 
#3 loops here. first one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_syn = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "synagogue"}
    params_encoded_syn = urlencode(params_syn)
    urlfinal_syn = f"{url}?{params_encoded_syn}"
    print(urlfinal_syn)
    r_final_1_syn = requests.get(urlfinal_syn)
    r_finaljson_1_syn = json.loads(r_final_1_syn.text)  
    synagogue_output_results.append(r_finaljson_1_syn)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_syn:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_syn = {"key": api_key,"pagetoken":r_finaljson_1_syn['next_page_token']}
        params_encoded_2_syn = urlencode(params_2_syn)
        urlfinal2_syn = f"{url}?{params_encoded_2_syn}"
        print(urlfinal2_syn)
        r_final_2_syn=requests.get(urlfinal2_syn)
        r_finaljson_2_syn = r_final_2_syn.json()
        synagogue_output_results.append(r_finaljson_2_syn)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_syn:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_syn = {"key": api_key,"pagetoken":r_finaljson_2_syn['next_page_token']}
                params_encoded_3_syn = urlencode(params_3_syn)
                urlfinal3_syn = f"{url}?{params_encoded_3_syn}"
                print(urlfinal3_syn)
                r_final_3_syn = requests.get(urlfinal3_syn)
                r_finaljson_3_syn = r_final_3_syn.json()
                synagogue_output_results.append(r_finaljson_3_syn)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/synagogue_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(synagogue_output_results, outfile, indent=4)
#test
len(synagogue_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(synagogue_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(synagogue_output_results[3]['results'][1]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
all_synagogues_london = []

def get_latlons (L):
    for everything in synagogue_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(all_synagogues_london)

len(all_synagogues_london)
type(all_synagogues_london)

print(all_synagogues_london)

#create a data  frame from the list
all_synagogues_london_df = pd.DataFrame(all_synagogues_london)
all_synagogues_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
all_synagogues_london_df_latlonextracted = pd.concat([all_synagogues_london_df.drop([1], axis=1), all_synagogues_london_df[1].apply(pd.Series)], axis=1)
all_synagogues_london_df_latlonextracted.head()

#remove duplicates
all_synagogues_london_df_latlonextracted_nodups = all_synagogues_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
all_synagogues_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/synagogues_output_results_DF_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
all_synagogues_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/synagogues_output_results_DF_latlonextract_nodups.csv")
all_synagogues_london_df_latlonextracted_nodups.head()

#create GDF using the lat and long coords for geom column.
all_synagogues_london_Gdf = gpd.GeoDataFrame(all_synagogues_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(all_synagogues_london_df_latlonextracted_nodups.lng, all_synagogues_london_df_latlonextracted_nodups.lat))
all_synagogues_london_Gdf['0']

all_synagogues_london_Gdf.crs

all_synagogues_london_Gdf.crs = {'init' :'epsg:4326'}
all_synagogues_london_Gdf.crs

all_synagogues_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_synagogues_london_Gdf.shp')


#%% API for temples


#create empty list for storing results 
temple_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid as a search point and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written to  a file. 
#3 loops here. first one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_temp = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "temple"}
    params_encoded_temp = urlencode(params_temp)
    urlfinal_temp = f"{url}?{params_encoded_temp}"
    print(urlfinal_temp)
    r_final_1_temp = requests.get(urlfinal_temp)
    r_finaljson_1_temp = json.loads(r_final_1_temp.text)  
    temple_output_results.append(r_finaljson_1_temp)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_temp:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_temp = {"key": api_key,"pagetoken":r_finaljson_1_temp['next_page_token']}
        params_encoded_2_temp = urlencode(params_2_temp)
        urlfinal2_temp = f"{url}?{params_encoded_2_temp}"
        print(urlfinal2_temp)
        r_final_2_temp=requests.get(urlfinal2_temp)
        r_finaljson_2_temp = r_final_2_temp.json()
        temple_output_results.append(r_finaljson_2_temp)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_temp:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_temp = {"key": api_key,"pagetoken":r_finaljson_2_temp['next_page_token']}
                params_encoded_3_temp = urlencode(params_3_temp)
                urlfinal3_temp = f"{url}?{params_encoded_3_temp}"
                print(urlfinal3_temp)
                r_final_3_temp = requests.get(urlfinal3_temp)
                r_finaljson_3_temp = r_final_3_temp.json()
                temple_output_results.append(r_finaljson_3_temp)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/temple_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(temple_output_results, outfile, indent=4)
#test
len(temple_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(temple_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(temple_output_results[3]['results'][1]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
all_temples_london = []

def get_latlons (L):
    for everything in temple_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(all_temples_london)

len(all_temples_london)
type(all_temples_london)

print(all_temples_london)

#create a data  frame from the list
all_temples_london_df = pd.DataFrame(all_temples_london)
all_temples_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
all_temples_london_df_latlonextracted = pd.concat([all_temples_london_df.drop([1], axis=1), all_temples_london_df[1].apply(pd.Series)], axis=1)
all_temples_london_df_latlonextracted.head()
all_temples_london_df_latlonextracted[0]

#remove duplicates
all_temples_london_df_latlonextracted_nodups = all_temples_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
all_temples_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/temples_output_results_DF_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
all_temples_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/temples_output_results_DF_latlonextract_nodups.csv")
all_temples_london_df_latlonextracted_nodups.head()
all_temples_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
all_temples_london_Gdf = gpd.GeoDataFrame(all_temples_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(all_temples_london_df_latlonextracted_nodups.lng, all_temples_london_df_latlonextracted_nodups.lat))
all_temples_london_Gdf['0']

#check CRS
all_temples_london_Gdf.crs
all_temples_london_Gdf.crs = {'init' :'epsg:4326'}
all_temples_london_Gdf.crs


all_temples_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_temples_london_Gdf.shp')


#%% API for wheelchair 

#create empty list for storing results 
wheelchair_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written a file. 
#3 loops here. one big one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_wheel = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "wheelchair"}
    params_encoded_wheel = urlencode(params_wheel)
    urlfinal_wheel = f"{url}?{params_encoded_wheel}"
    print(urlfinal_wheel)
    r_final_1_wheel = requests.get(urlfinal_wheel)
    r_finaljson_1_wheel = json.loads(r_final_1_wheel.text)  
    wheelchair_output_results.append(r_finaljson_1_wheel)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_wheel:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_wheel = {"key": api_key,"pagetoken":r_finaljson_1_wheel['next_page_token']}
        params_encoded_2_wheel = urlencode(params_2_wheel)
        urlfinal2_wheel = f"{url}?{params_encoded_2_wheel}"
        print(urlfinal2_wheel)
        r_final_2_wheel = requests.get(urlfinal2_wheel)
        r_finaljson_2_wheel = r_final_2_wheel.json()
        wheelchair_output_results.append(r_finaljson_2_wheel)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_wheel:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_wheel = {"key": api_key,"pagetoken":r_finaljson_2_wheel['next_page_token']}
                params_encoded_3_wheel = urlencode(params_3_wheel)
                urlfinal3_wheel = f"{url}?{params_encoded_3_wheel}"
                print(urlfinal3_wheel)
                r_final_3_wheel = requests.get(urlfinal3_wheel)
                r_finaljson_3_wheel = r_final_3_wheel.json()
                wheelchair_output_results.append(r_finaljson_3_wheel)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/wheelchair_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(wheelchair_output_results, outfile, indent=4)


#test
len(wheelchair_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(wheelchair_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(wheelchair_output_results[3]['results'][1]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
wheelchairs_london = []

def get_latlons (L):
    for everything in wheelchair_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(wheelchairs_london)

len(wheelchairs_london)
type(wheelchairs_london)

print(wheelchairs_london)

#create a data  frame from the list
wheelchairs_london_df = pd.DataFrame(wheelchairs_london)
wheelchairs_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
wheelchairs_london_df_latlonextracted = pd.concat([wheelchairs_london_df.drop([1], axis=1), wheelchairs_london_df[1].apply(pd.Series)], axis=1)
wheelchairs_london_df_latlonextracted.head()
wheelchairs_london_df_latlonextracted[0]

#remove duplicates
wheelchairs_london_df_latlonextracted_nodups = wheelchairs_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
wheelchairs_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/wheelchairs_london_df_latlonextract_nodups.csv")


#read in as csv to make columns string for writnig to shapefile 
wheelchairs_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/wheelchairs_london_df_latlonextract_nodups.csv")
wheelchairs_london_df_latlonextracted_nodups.head()
wheelchairs_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
wheelchairs_london_Gdf = gpd.GeoDataFrame(wheelchairs_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(wheelchairs_london_df_latlonextracted_nodups.lng, wheelchairs_london_df_latlonextracted_nodups.lat))
wheelchairs_london_Gdf['0']

wheelchairs_london_Gdf.crs

wheelchairs_london_Gdf.crs = {'init' :'epsg:4326'}
wheelchairs_london_Gdf.crs

wheelchairs_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/wheelchairs_london_Gdf.shp')




#%% API for disabled 

#create empty list for storing results 
disabled_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written a file. 
#3 loops here. one big one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_disabled = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "disabled"}
    params_encoded_disabled = urlencode(params_disabled)
    urlfinal_disabled = f"{url}?{params_encoded_disabled}"
    print(urlfinal_disabled)
    r_final_1_disabled = requests.get(urlfinal_disabled)
    r_finaljson_1_disabled = json.loads(r_final_1_disabled.text)  
    disabled_output_results.append(r_finaljson_1_disabled)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_disabled:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_disabled = {"key": api_key,"pagetoken":r_finaljson_1_disabled['next_page_token']}
        params_encoded_2_disabled = urlencode(params_2_disabled)
        urlfinal2_disabled = f"{url}?{params_encoded_2_disabled}"
        print(urlfinal2_disabled)
        r_final_2_disabled = requests.get(urlfinal2_disabled)
        r_finaljson_2_disabled = r_final_2_disabled.json()
        disabled_output_results.append(r_finaljson_2_disabled)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_disabled:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_disabled = {"key": api_key,"pagetoken":r_finaljson_2_disabled['next_page_token']}
                params_encoded_3_disabled = urlencode(params_3_disabled)
                urlfinal3_disabled = f"{url}?{params_encoded_3_disabled}"
                print(urlfinal3_disabled)
                r_final_3_disabled = requests.get(urlfinal3_disabled)
                r_finaljson_3_disabled = r_final_3_disabled.json()
                disabled_output_results.append(r_finaljson_3_disabled)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/disabled_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(disabled_output_results, outfile, indent=4)


#test
len(disabled_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(disabled_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(disabled_output_results[1]['results'][3]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
disabled_london = []

def get_latlons (L):
    for everything in disabled_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(disabled_london)

len(disabled_london)
type(disabled_london)

print(disabled_london)

#create a data  frame from the list
disabled_london_df = pd.DataFrame(disabled_london)
disabled_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
disabled_london_df_latlonextracted = pd.concat([disabled_london_df.drop([1], axis=1), disabled_london_df[1].apply(pd.Series)], axis=1)
disabled_london_df_latlonextracted.head()
disabled_london_df_latlonextracted[0]

#remove duplicates
disabled_london_df_latlonextracted_nodups = disabled_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
disabled_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/disabled_london_df_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
disabled_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/disabled_london_df_latlonextract_nodups.csv")
disabled_london_df_latlonextracted_nodups.head()
disabled_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
disabled_london_Gdf = gpd.GeoDataFrame(disabled_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(disabled_london_df_latlonextracted_nodups.lng, disabled_london_df_latlonextracted_nodups.lat))
disabled_london_Gdf['0']

disabled_london_Gdf.crs

disabled_london_Gdf.crs = {'init' :'epsg:4326'}
disabled_london_Gdf.crs

disabled_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/disabled_london_Gdf.shp')



#%% API for community centre 

#create empty list for storing results 
community_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written a file. 
#3 loops here. one big one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_community = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "community centre"}
    params_encoded_community = urlencode(params_community)
    urlfinal_community = f"{url}?{params_encoded_community}"
    print(urlfinal_community)
    r_final_1_community = requests.get(urlfinal_community)
    r_finaljson_1_community = json.loads(r_final_1_community.text)  
    community_output_results.append(r_finaljson_1_community)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_community:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_community = {"key": api_key,"pagetoken":r_finaljson_1_community['next_page_token']}
        params_encoded_2_community = urlencode(params_2_community)
        urlfinal2_community = f"{url}?{params_encoded_2_community}"
        print(urlfinal2_community)
        r_final_2_community = requests.get(urlfinal2_community)
        r_finaljson_2_community = r_final_2_community.json()
        community_output_results.append(r_finaljson_2_community)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_community:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_community = {"key": api_key,"pagetoken":r_finaljson_2_community['next_page_token']}
                params_encoded_3_community = urlencode(params_3_community)
                urlfinal3_community = f"{url}?{params_encoded_3_community}"
                print(urlfinal3_community)
                r_final_3_community = requests.get(urlfinal3_community)
                r_finaljson_3_community = r_final_3_community.json()
                community_output_results.append(r_finaljson_3_community)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/community_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(community_output_results, outfile, indent=4)


#test
len(community_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(community_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(community_output_results[1]['results'][3]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
community_london = []

def get_latlons (L):
    for everything in community_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(community_london)

len(community_london)
type(community_london)

print(community_london)

#create a data  frame from the list
community_london_df = pd.DataFrame(community_london)
community_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
community_london_df_latlonextracted = pd.concat([community_london_df.drop([1], axis=1), community_london_df[1].apply(pd.Series)], axis=1)
community_london_df_latlonextracted.head()
community_london_df_latlonextracted[0]

#remove duplicates
community_london_df_latlonextracted_nodups = community_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
community_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/community_london_df_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
community_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/community_london_df_latlonextract_nodups.csv")
community_london_df_latlonextracted_nodups.head()
community_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
community_london_Gdf = gpd.GeoDataFrame(community_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(community_london_df_latlonextracted_nodups.lng, community_london_df_latlonextracted_nodups.lat))
community_london_Gdf['0']

community_london_Gdf.crs

community_london_Gdf.crs = {'init' :'epsg:4326'}
community_london_Gdf.crs


community_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/community_london_Gdf.shp')



#%% API for LGBTQ 
#create empty list for storing results 
LGBT_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written a file. 
#3 loops here. one big one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_LGBT = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "LGBT"}
    params_encoded_LGBT = urlencode(params_LGBT)
    urlfinal_LGBT = f"{url}?{params_encoded_LGBT}"
    print(urlfinal_LGBT)
    r_final_1_LGBT = requests.get(urlfinal_LGBT)
    r_finaljson_1_LGBT = json.loads(r_final_1_LGBT.text)  
    LGBT_output_results.append(r_finaljson_1_LGBT)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_LGBT:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_LGBT = {"key": api_key,"pagetoken":r_finaljson_1_LGBT['next_page_token']}
        params_encoded_2_LGBT = urlencode(params_2_LGBT)
        urlfinal2_LGBT = f"{url}?{params_encoded_2_LGBT}"
        print(urlfinal2_LGBT)
        r_final_2_LGBT = requests.get(urlfinal2_LGBT)
        r_finaljson_2_LGBT = r_final_2_LGBT.json()
        LGBT_output_results.append(r_finaljson_2_LGBT)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_LGBT:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_LGBT = {"key": api_key,"pagetoken":r_finaljson_2_LGBT['next_page_token']}
                params_encoded_3_LGBT = urlencode(params_3_LGBT)
                urlfinal3_LGBT = f"{url}?{params_encoded_3_LGBT}"
                print(urlfinal3_LGBT)
                r_final_3_LGBT = requests.get(urlfinal3_LGBT)
                r_finaljson_3_LGBT = r_final_3_LGBT.json()
                LGBT_output_results.append(r_finaljson_3_LGBT)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/LGBT_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(LGBT_output_results, outfile, indent=4)


#test
len(LGBT_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(LGBT_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(LGBT_output_results[1]['results'][3]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
LGBT_london = []

def get_latlons (L):
    for everything in LGBT_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(LGBT_london)

len(LGBT_london)
type(LGBT_london)

print(LGBT_london)

#create a data  frame from the list
LGBT_london_df = pd.DataFrame(LGBT_london)
LGBT_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
LGBT_london_df_latlonextracted = pd.concat([LGBT_london_df.drop([1], axis=1), LGBT_london_df[1].apply(pd.Series)], axis=1)
LGBT_london_df_latlonextracted.head()
LGBT_london_df_latlonextracted[0]

#remove duplicates
LGBT_london_df_latlonextracted_nodups = LGBT_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
LGBT_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/LGBT_london_df_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
LGBT_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/LGBT_london_df_latlonextract_nodups.csv")
LGBT_london_df_latlonextracted_nodups.head()
LGBT_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
LGBT_london_Gdf = gpd.GeoDataFrame(LGBT_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(LGBT_london_df_latlonextracted_nodups.lng, LGBT_london_df_latlonextracted_nodups.lat))
LGBT_london_Gdf['0']

LGBT_london_Gdf.crs

LGBT_london_Gdf.crs = {'init' :'epsg:4326'}
LGBT_london_Gdf.crs

LGBT_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/LGBT_london_Gdf.shp')



#%% API for transgender 

#create empty list for storing results 
transgender_output_results = []

#loop for the API requests from google maps api. this uses the borough centroid and searches at a radius previously
#calculated for all instances matching the key word. 
#results are then stored in the output list and written a file. 
#3 loops here. one big one to go through all borough centroids. 2 nested loops to iterate through additional results pages
for latlon in boroughcentroids_string:
    params_transgender = {"key": api_key,"location":latlon,"radius": 6000,"keyword": "transgender"}
    params_encoded_transgender = urlencode(params_transgender)
    urlfinal_transgender = f"{url}?{params_encoded_transgender}"
    print(urlfinal_transgender)
    r_final_1_transgender = requests.get(urlfinal_transgender)
    r_finaljson_1_transgender = json.loads(r_final_1_transgender.text)  
    transgender_output_results.append(r_finaljson_1_transgender)
    print("appending 1st page to the list")

    if 'next_page_token' in r_finaljson_1_transgender:
        print("yes 2nd page token found")
        #sleep to wait for token to load
        time.sleep(3)
        params_2_transgender = {"key": api_key,"pagetoken":r_finaljson_1_transgender['next_page_token']}
        params_encoded_2_transgender = urlencode(params_2_transgender)
        urlfinal2_transgender = f"{url}?{params_encoded_2_transgender}"
        print(urlfinal2_transgender)
        r_final_2_transgender = requests.get(urlfinal2_transgender)
        r_finaljson_2_transgender = r_final_2_transgender.json()
        transgender_output_results.append(r_finaljson_2_transgender)
        print('2nd page appending to list')
        
        if 'next_page_token' in r_finaljson_2_transgender:
                print("yes final page token found")
                #sleep to wait for token to load
                time.sleep(3)
                params_3_transgender = {"key": api_key,"pagetoken":r_finaljson_2_transgender['next_page_token']}
                params_encoded_3_transgender = urlencode(params_3_transgender)
                urlfinal3_transgender = f"{url}?{params_encoded_3_transgender}"
                print(urlfinal3_transgender)
                r_final_3_transgender = requests.get(urlfinal3_transgender)
                r_finaljson_3_transgender = r_final_3_transgender.json()
                transgender_output_results.append(r_finaljson_3_transgender)
                print('3rd and final page appending to list')
        else: 
                print("no next page token found") 
        
    else: 
        print("no second page token found")
              
    with open('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/transgender_output_results_max_next_page_tok.txt', 'w') as outfile:
            json.dump(transgender_output_results, outfile, indent=4)


#test
len(transgender_output_results)
#print results to check output
#the output is a json (stored as dictionary) inside a list
print(transgender_output_results)

#specific data can be accessed as follows by accessing the specific list postion, then using dictionary keys
print(transgender_output_results[0]['results'][1]['name'])


#function to get all names,lat,longs from all mosques found in the API results.  

#blank list for results
transgender_london = []

def get_latlons (L):
    for everything in transgender_output_results:
        results_accessed = everything['results']
        for resultsloop in results_accessed:
            location = resultsloop['geometry']['location']
            name = resultsloop['name']
            print(name, location)
            L.append([name, location])
            #L.append(location) 

get_latlons(transgender_london)

len(transgender_london)
type(transgender_london)

print(transgender_london)

#create a data  frame from the list
transgender_london_df = pd.DataFrame(transgender_london)
transgender_london_df.head()
#extract the lat and longs into separate columns ready to make a geom column for gdf. this code drops the original column and creates 2 new ones using the apply(pd.Series)
transgender_london_df_latlonextracted = pd.concat([transgender_london_df.drop([1], axis=1), transgender_london_df[1].apply(pd.Series)], axis=1)
transgender_london_df_latlonextracted.head()
transgender_london_df_latlonextracted[0]

#remove duplicates
transgender_london_df_latlonextracted_nodups = transgender_london_df_latlonextracted.drop_duplicates(subset = 0, keep = 'first') 
transgender_london_df_latlonextracted_nodups.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/transgender_london_df_latlonextract_nodups.csv")




#read in as csv to make columns string for writnig to shapefile 
transgender_london_df_latlonextracted_nodups = pd.read_csv("C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/GoogleMapsAPI/transgender_london_df_latlonextract_nodups.csv")
transgender_london_df_latlonextracted_nodups.head()
transgender_london_df_latlonextracted_nodups['0']

#create GDF using the lat and long coords for geom column.
transgender_london_Gdf = gpd.GeoDataFrame(transgender_london_df_latlonextracted_nodups, geometry=gpd.points_from_xy(transgender_london_df_latlonextracted_nodups.lng, transgender_london_df_latlonextracted_nodups.lat))
transgender_london_Gdf['0']

transgender_london_Gdf.crs

transgender_london_Gdf.crs = {'init' :'epsg:4326'}
transgender_london_Gdf.crs

transgender_london_Gdf.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/transgender_london_Gdf.shp')
