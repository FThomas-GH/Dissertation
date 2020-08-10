#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 09:59:05 2020

@author: freddythomas
"""
#import libraries

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


#read in LSOA boundary file from London data store
lsoa = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp')
#check coordinate reference system. LSOA imported as BNG need to convert to WGS84 for Google. 
lsoa.crs
#set CRS from the data. 
lsoa = gpd.GeoDataFrame(lsoa,geometry='geometry',crs={'init':'epsg:27700'})
#re-project data into WGS84
lsoa = lsoa.to_crs("EPSG:4326")

#checks
lsoa.crs
list(lsoa.columns)
type(lsoa)

#%%
#read in the entire excel workbook with multple sheets. 
crimes_wb = pd.ExcelFile('C:/Users/tho84231/Documents/GitHub/Dissertation/MockData.xlsx')

#get list of sheet names to parse all sheets into data frames to then be merged. 
crimes_wb_sheetnames = crimes_wb.sheet_names
print(crimes_wb_sheetnames)
#parse sheet names to make seperate data frames for each hate crime incident
crimes_A = crimes_wb.parse('Anti Semitic Incidents')
crimes_D = crimes_wb.parse('Disability Hate Incidents')
crimes_F = crimes_wb.parse('Faith Hate incidents')
crimes_H = crimes_wb.parse('Homophobic Hate Incidents')
crimes_I = crimes_wb.parse('Islamophobic Hate Incidents')
crimes_R = crimes_wb.parse('Racist Hate Incidents')
crimes_T = crimes_wb.parse('Transgender Hate Incidents')

#check
crimes_A
crimes_D
crimes_F
crimes_T
crimes_H 
#%% 
#add column for type of hate crime to keep information clear when joined. 
list(crimes_A.columns)
crimes_A['Type']='Anti Semitic'
crimes_D['Type']='Disability'
crimes_F['Type']='Faith'
crimes_H['Type']='Homophobic'
crimes_I['Type']='Islamophobic'
crimes_R['Type']='Racist'
crimes_T['Type']='Transgender'

list(crimes_A.columns)
#check
crimes_A['Type'].tail()
crimes_T['Type'].tail()


crimes_T['LSOA'].head(20)

#concat all data frames together. add keys to keep track of where data have come from.  
frames = [crimes_A, crimes_D, crimes_F, crimes_H, crimes_I, crimes_R, crimes_T]
hatecrimes = pd.concat(frames, keys=['Anti Semitic', 'Disability', 'Faith', 'Homophobic', 'Islamophobic', 'Racist', 'Transgender'])

hatecrimes.head()

#check
print(type(hatecrimes)) 

#%%

#replace '(blank)' that is written in some of the rows in the LSOA column. This is written for crimes that had no LSOA attributed to them 
hatecrimes['LSOA'].replace('(blank)', 'No LSOA', inplace=True)

#replace blank values (NaN) with the value preceeding it to fill out LSOA column. This will ensure all rows of hate crime incidents have an LSOA code.
#there are blanks in the data because when multiple incidents occur in an LSOA the raw data did not copy the LSOA code into the following rows. 
hatecrimes['LSOA']= hatecrimes['LSOA'].fillna(method='ffill')
#check
hatecrimes.tail(60)

#drop rows that contain total as the data are duplicated for each LSOA to create final dataframe and label as just df for simplicity. 
df = hatecrimes[hatecrimes["LSOA"].str.contains('Total')==False]
df['LSOA']

#check
df.tail(40)


#save. 
df.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/allhatecrimesmerged.csv')
print(type(df))


#%% Reshaping and slicing of data. 

list(df)
df.columns


#total hate crime incidents across all hate crime categories together and all months per LSOA. Add to new named column
totalhatecrimesLSOA = df.groupby('LSOA')['Grand Total'].sum().reset_index(name ='Total Incidents')
totalhatecrimesLSOA.head(10)
#sort by desecnding to see the LSOA's that have the highest number of overall hate crimes
totalhatecrimesLSOA.sort_values(by=['Total Incidents'], ascending=False).head(20)


#analyse the total hate crimes for each category of hate crime separately per LSOA. 
totalhatecrimesLSOA_by_type = df.groupby(['LSOA', 'Type'])['Grand Total'].sum().reset_index(name ='Total Incidents')
totalhatecrimesLSOA_by_type.head(10)
totalhatecrimesLSOA_by_type.sort_values(by=['Total Incidents'], ascending=False).head(20)


#a quick check that the slicing has worked correctly. The total crimes by type summed together should = total crimes above totals  
totalhatecrimesLSOA.set_index('LSOA').loc['E01000006']
totalhatecrimesLSOA_by_type.set_index('LSOA').loc['E01000006']
#confirm working correctly. 


#totalhatecrimesLSOA_by_type.set_index('LSOA')

#reshape the data so that each LSOA is the index, with the hate crime categories as columns. 
totalhatecrimesLSOA_by_type_clean = totalhatecrimesLSOA_by_type.set_index('LSOA').pivot(columns='Type', values='Total Incidents')
totalhatecrimesLSOA_by_type_clean
#create a new column for total hate crimes for each LSOA
totalhatecrimesLSOA_by_type_clean['Total All Incidents'] = totalhatecrimesLSOA_by_type_clean.sum(axis=1)

#check the results. for LSOA E01000006 the value for total incidencts should be 253 as calculated above. 
totalhatecrimesLSOA_by_type_clean.head()

#export to csv to save it and reset index so lsoa column is still maintained. 
totalhatecrimesLSOA_by_type_clean.reset_index().to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/lsoa_and_hatecrimes_clean.csv')



#%% FAITH


#check LSOA's have correct data with raw data (This works for the real data, however for the mock data the totals do not match as outlined in the READ ME)
totalhatecrimesLSOA_by_type_clean.loc['E01000123']

#get totals for each category of hate crime and total incidents. 
totalhatecrimesLSOA_by_type_clean.sum()


#Remove Islamophobic hate crimes and Anti-Semitic hate crimes from the Faith hate crime catefory to remove double counting  
#list columns
list(totalhatecrimesLSOA_by_type_clean)
#create new column that sums together both anti semitic column and islamophobic hate crime values. 
totalhatecrimesLSOA_by_type_clean['AntiSem_and_Islam'] =  totalhatecrimesLSOA_by_type_clean[['Anti Semitic', 'Islamophobic']].sum(axis=1)

#take away these values from the original Faith hate crime values. 
totalhatecrimesLSOA_by_type_clean['Faith_noIslamAntiSem'] = totalhatecrimesLSOA_by_type_clean['Faith'] - totalhatecrimesLSOA_by_type_clean['AntiSem_and_Islam']
#check the total for the new Faith hate crimes column without Anti Semitic and Islamophobic incidents. All Faith values should stil be 0 or above
totalhatecrimesLSOA_by_type_clean['Faith_noIslamAntiSem'].sum()
#due to the random generation of numbers for the mock data, this number is a negative value. However, from the real data it was not. 

#test to see worked but checking the sums worked. 
totalhatecrimesLSOA_by_type_clean.loc['E01000010']
totalhatecrimesLSOA_by_type_clean.loc['E01000307']

totalhatecrimesLSOA_by_type_clean.sort_values(by=['Faith_noIslamAntiSem'], ascending=False).head(20)

totalhatecrimesLSOA_by_type_clean.sum()

#check against raw data. But this does not work with the mock data.  
totalhatecrimesLSOA_by_type_clean.loc['E01004259']


list(totalhatecrimesLSOA_by_type_clean)

#replace NaN values with 0's for analysis. 
totalhatecrimesLSOA_by_type_clean_nonNAN = totalhatecrimesLSOA_by_type_clean.replace(np.nan,0)

totalhatecrimesLSOA_by_type_clean_nonNAN.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/totalhatecrimesLSOA_by_type_clean_noNAN.csv')



#%%
#THIS IS WHERE THE EXPLORATORY DATA ANALYSIS STARTS. 

#PLOT bar to see totals compared of each 
list(totalhatecrimesLSOA_by_type_clean_nonNAN)


hatecrimes_describe = totalhatecrimesLSOA_by_type_clean_nonNAN.describe()
hatecrimes_describe
hatecrimes_describe.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/hatecrimes_clean_describe.csv')


#boxplot everything
list(totalhatecrimesLSOA_by_type_clean_nonNAN)

#box plot all hate crimes together. 
ax = sns.boxplot(data=totalhatecrimesLSOA_by_type_clean_nonNAN, orient="h", palette="Set2")
ax.figsize=(30,30)
ax.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/BoxPlotofAll.png')


#closer inspection of all box plots
AS_bp = sns.boxplot(x='Anti Semitic', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
AS_bp.set_title('Boxplot for Anti Semitic Hate Crime')
AS_bp.figure.figsize=(15,15)
AS_bp.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/AS_bp.png')


D_bp = sns.boxplot(x='Disability', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
D_bp.set_title('Boxplot for Disability Hate Crime')
D_bp.figure.figsize=(15,15)
D_bp.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/D_bp.png')


F_bp = sns.boxplot(x='Faith', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
F_bp.set_title('Boxplot for Faith Hate Crime')
F_bp.figure.figsize=(15,15)
F_bp.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/F_bp.png')



H_bp = sns.boxplot(x='Homophobic', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
H_bp.set_title('Boxplot for Homophobic Hate Crime')
H_bp.figure.figsize=(15,15)
H_bp.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/H_bp.png')


#check correlations
correlations = totalhatecrimesLSOA_by_type_clean_nonNAN.corr()

matrix = np.triu(correlations)
sns.heatmap(correlations, annot=True, mask=matrix)

hatecrimecor = sns.heatmap(correlations, annot=True, cmap='GnBu', square=True)
hatecrimecor.figure.savefig('C:/Users/tho84231/Documents/GitHub/Dissertation/HatecrimeCorr.png')



#%%
#HISTOGRAMS
#plot and explore each category
#distribution plot for each variable and all together
#drop NO LSOA AS IT MESSES UP THE DIST PLOTS

#EXAMINE THE NAN row values
pd.set_option('display.max_columns', None)
print(totalhatecrimesLSOA_by_type_clean_nonNAN.tail())
totalhatecrimesLSOA_by_type_clean_nonNAN_NoLSOA = totalhatecrimesLSOA_by_type_clean_nonNAN.iloc[[0, -1]]
print(totalhatecrimesLSOA_by_type_clean_nonNAN_NoLSOA)
pd.reset_option('max_columns')



#drop No LSOA from LSOA codes. Which is the last row
totalhatecrimesLSOA_by_type_clean_nonNAN.shape
totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa = totalhatecrimesLSOA_by_type_clean_nonNAN[:-1]
totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.shape

totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.tail()

totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.sort_values(by=['Total All Incidents'], ascending=False).head(20)


totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.columns


#CLEAN data set  FOR SCATTERPLOT with only columns needed and better column tiles 

finalhatedataset = totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa[['Anti Semitic','Disability','Homophobic','Islamophobic','Racist','Transgender','Faith_noIslamAntiSem', 'Total All Incidents']]
finalhatedataset
#rename columns for graph
finalhatedataset.rename(columns={"Anti_Semitic": "Anti-Semitic",'Faith_noIslamAntiSem':'Faith'}, inplace=True)
finalhatedataset.sum()
finalhatedataset.reset_index().to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/FinalHateCrimeDataset.csv')

#scatterplot and distbrution plot for all variables together
#pair plot. 
pair = sns.pairplot(finalhatedataset, kind='scatter',height=1.2)
#pair plot with refression line
pairreg = sns.pairplot(finalhatedataset, kind='reg',height=1.2)





#or to look at all box plot distribution at once. 
ax = sns.boxplot(data=finalhatedataset, orient='h')



#as there seems to be lots of outliers we can test for this.

#create IQR for each column to see out
finalhatedataset
IQR_df = finalhatedataset.copy()
IQR_df.sum()
Q1 = IQR_df.quantile(0.25)
Q3 = IQR_df.quantile(0.75)
iqr = Q3-Q1
print(iqr)

print(IQR_df < (Q1 - 1.5 * iqr )) or (IQR_df > (Q3 + 1.5 * iqr))

outliertruefalse = print(IQR_df < (Q1 - 1.5 * iqr )) or (IQR_df > (Q3 + 1.5 * iqr))
outliertruefalse = outliertruefalse.reset_index()
outliertruefalse.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/outliers_truefalse.csv')


#check there is a row for an LSOA for each hate crime type
outliertruefalse.loc[melted_outlierstruefalse['LSOA'] =='E01000002']





#%% join hate crime data to LSOA shapefile

lsoa.crs
list(finalhatedataset)
list(lsoa)
finalhatedataset.reset_index()
lsoa_with_hatecrimes = pd.merge(lsoa,finalhatedataset, how='outer', left_on='LSOA11CD', right_on='LSOA')

lsoa_with_hatecrimes.columns

#save as shapefile and  geojson 
lsoa_with_hatecrimes.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime.shp')
lsoa_with_hatecrimes.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime.geojson', driver='GeoJSON')




#%% add in population for 2018

lsoa_2018pop = pd.read_excel('C:/Users/tho84231/Documents/GitHub/Dissertation/LSOA 2018pop estimates/SAPE21DT1a-mid-2018-on-2019-LA-lsoa-syoa-estimates-formatted.xlsx', sheet_name='Mid-2018 Persons', skiprows=3)
lsoa_2018pop.head()

#rename first row to be column headers and drop the first row from the data 
lsoa_2018pop.rename(columns=lsoa_2018pop.iloc[0], inplace = True)
lsoa_2018pop.drop([0], inplace = True)

lsoa_2018pop.head()

list(lsoa_2018pop)

#as the data contains populations per age, the data is slimmed down to provide only the total population per LSOA. 
lsoa_2018pop_allage = lsoa_2018pop[['Area Codes', 'LSOA','LA (2019 boundaries)', 'All Ages']]
list(lsoa_2018pop_allage)

type(lsoa)
type(lsoa_2018pop_allage)

list(lsoa_with_hatecrimes)
list(lsoa_2018pop_allage)

lsoa_2018pop_allage
#merge the existing LSOA and hate crime gdf with population  LSOA data. 
lsoa_with_hatecrimes_and_pop = pd.merge(lsoa_with_hatecrimes,lsoa_2018pop_allage, how='outer', left_on='LSOA11CD', right_on='Area Codes')
#check merged correctly
list(lsoa_with_hatecrimes_and_pop) 


#calculate crime rate per 1000
list(lsoa_with_hatecrimes_and_pop)

lsoa_with_hatecrimes_and_pop['AntiSem_p1000'] =lsoa_with_hatecrimes_and_pop['Anti Semitic'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Disability_p1000'] =lsoa_with_hatecrimes_and_pop['Disability'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Homophobic_p1000'] =lsoa_with_hatecrimes_and_pop['Homophobic'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Islamophobic_p1000'] =lsoa_with_hatecrimes_and_pop['Islamophobic'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Racist_p1000'] =lsoa_with_hatecrimes_and_pop['Racist'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Transgender_p1000'] =lsoa_with_hatecrimes_and_pop['Transgender'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['Faith_p1000'] =lsoa_with_hatecrimes_and_pop['Faith'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000
lsoa_with_hatecrimes_and_pop['All_HC_rate_p1000'] =lsoa_with_hatecrimes_and_pop['Total All Incidents'] / lsoa_with_hatecrimes_and_pop['All Ages']*1000

#check
list(lsoa_with_hatecrimes_and_pop)

#examine top LSOA 
lsoa_with_hatecrimes_and_pop.sort_values(by=['All_HC_rate_p1000'], ascending=False).head(20)

#remove all white space in columns
lsoa_with_hatecrimes_and_pop.columns = lsoa_with_hatecrimes_and_pop.columns.str.replace(' ', '_')

list(lsoa_with_hatecrimes_and_all_data)
type(lsoa_with_hatecrimes_and_all_data)

#export final LSOA geodataframe to files for use in Arc. 
lsoa_with_hatecrimes_and_pop.crs
lsoa_with_hatecrimes_and_pop.to_csv("C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime_and_population.csv")
lsoa_with_hatecrimes_and_pop.to_file("C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime_and_population.shp")
lsoa_with_hatecrimes_and_pop.to_file("C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime_and_population.geojson", driver='GeoJSON')
