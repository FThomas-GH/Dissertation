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
hatecrimes_describe.to_csv('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/hatecrimes_clean_describe.csv')


#null values?
null = totalhatecrimesLSOA_by_type_clean.isna().sum()/len(df)
null[null > 0].sort_values()
nullpercent = null*100
nullpercent.sort_values(ascending=False)

#boxplot everything
list(totalhatecrimesLSOA_by_type_clean_nonNAN)

ax = sns.boxplot(data=totalhatecrimesLSOA_by_type_clean_nonNAN, orient="h", palette="Set2")
ax.figsize=(30,30)
ax.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\LSOAHateCrimeExploratoryAnalysis\BoxPlotofAll.png')


#closer inspection of all box plots
AS_bp = sns.boxplot(x='Anti Semitic', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
AS_bp.set_title('Boxplot for Anti Semitic Hate Crime')
AS_bp.figure.figsize=(15,15)
AS_bp.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\LSOAHateCrimeExploratoryAnalysis\AS_bp.png')


D_bp = sns.boxplot(x='Disability', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
D_bp.set_title('Boxplot for Disability Hate Crime')
D_bp.figure.figsize=(15,15)
D_bp.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\LSOAHateCrimeExploratoryAnalysis\D_bp.png')


F_bp = sns.boxplot(x='Faith', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
F_bp.set_title('Boxplot for Faith Hate Crime')
F_bp.figure.figsize=(15,15)
F_bp.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\LSOAHateCrimeExploratoryAnalysis\F_bp.png')



H_bp = sns.boxplot(x='Homophobic', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
H_bp.set_title('Boxplot for Homophobic Hate Crime')
H_bp.figure.figsize=(15,15)
H_bp.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\LSOAHateCrimeExploratoryAnalysis\H_bp.png')


#potentially do close ups for the rest of the plots
I_bp = sns.boxplot(x='Islamophobic', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
R_bp = sns.boxplot(x='Racist', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
T_bp = sns.boxplot(x='Transgender', data=totalhatecrimesLSOA_by_type_clean_nonNAN)
Tot_bp = sns.boxplot(x='Total All Incidents', data=totalhatecrimesLSOA_by_type_clean_nonNAN)



#check correlations
#TEMPORARY READ IN
import pandas as pd
import seaborn as sns
import numpy as np
finalhatedataset = pd.read_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\FINALHATEDATA.csv')
list(finalhatedataset)
finalhatedataset.set_index('LSOA11CD', inplace = True)
finalhatedataset.drop('Unnamed: 0', axis=1, inplace = True)
finalhatedataset

correlations = finalhatedataset.corr()

matrix = np.triu(correlations)
sns.heatmap(correlations, annot=True, mask=matrix)

hatecrimecor = sns.heatmap(correlations, annot=True, cmap='GnBu', square=True)
plt.tight_layout()
hatecrimecor.figure.figsize=(100,100)
hatecrimecor.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Graphs and Figures\HatecrimeCorr.png')


hatecrimes_nototal = totalhatecrimesLSOA_by_type_clean_nonNAN.drop('Total All Incidents', axis=1)

#HISTOGRAMS
#plot and explore each category
#distribution plot for each variable and all together
#drop NO LSOA AS IT MESSES UP THE DIST PLOTS

totalhatecrimesLSOA_by_type_clean_nonNAN.sort_values(by=['Total_All_Incidents'], ascending=False).head(20)
totalhatecrimesLSOA_by_type_clean_nonNAN.reset_index()

#EXAMINE THE NAN row values
pd.set_option('display.max_columns', None)
print(totalhatecrimesLSOA_by_type_clean_nonNAN.tail())
totalhatecrimesLSOA_by_type_clean_nonNAN_NANonly = totalhatecrimesLSOA_by_type_clean_nonNAN.iloc[[0, -1]]
print(totalhatecrimesLSOA_by_type_clean_nonNAN_NANonly)
pd.reset_option('max_columns')



#drop NAN from LSOA codes
totalhatecrimesLSOA_by_type_clean_nonNAN.reset_index()

totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa = totalhatecrimesLSOA_by_type_clean_nonNAN.reset_index().dropna().set_index('LSOA11CD')

totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.sort_values(by=['Total_All_Incidents'], ascending=False).head(20)


totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.columns


#CLEAN FOR SCATTERPLOT

finalhatedataset = totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa[['Anti_Semitic','Disability','Homophobic','Islamophobic','Racist','Transgender','Faith_noIslamAntiSem', 'Total_All_Incidents']]
finalhatedataset
finalhatedataset.rename(columns={"Anti_Semitic": "Anti Semitic",'Total_All_Incidents':'Total All Incidents','Faith_noIslamAntiSem':'Faith'}, inplace=True)
finalhatedataset.sum()
finalhatedataset.reset_index().to_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\FINALHATEDATA.csv')

#scatterplot and distbrution plot for all variables together

pair = sns.pairplot(finalhatedataset, kind='scatter',height=1.2)
pairreg = sns.pairplot(finalhatedataset, kind='reg',height=1.2)

#scatter matrix with no total column
pair_nototal = sns.pairplot(hatecrimes_nototal, kind='scatter',height=1.2)


#sliced to make them fit better. 
pairslices = sns.pairplot(totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa[['Anti Semitic', 'Disability', 'Faith', 'Homophobic']], plot_kws={'color':'green'}, kind='reg')


totalhatecrimesLSOA_by_type_clean_nomissinglsoa.columns


#individual distribution histograms close up. 
ax = sns.distplot(totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa['Transgender'], color='teal')


#old way of doing dist plots. 

#total_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Total All Incidents'].values
#r_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Racist'].values
#as_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Anti Semitic'].values
#d_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Disability'].values
#f_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Faith'].values
#h_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Homophobic'].values
#i_values = totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Islamophobic'].values
#t_values = totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa['Transgender'].values


#ax = sns.distplot(total_values, color='red')
#ax = sns.distplot(r_values, color='blue')
#ax = sns.distplot(as_values, color='green')
#ax = sns.distplot(d_values, color='grey')
#ax = sns.distplot(f_values, color='orange')
#ax = sns.distplot(h_values, color='cyan')
#ax = sns.distplot(i_values, color='purple')
#ax = sns.distplot(t_values, color='blue')



#this shows the data is heavily skewed right. as the mean will be on the right hand side. 


#box plot
totalhatecrimesLSOA_by_type_clean_nomissinglsoa.columns

#this does the same as below without needing new values. 
sns.boxplot(totalhatecrimesLSOA_by_type_clean_nomissinglsoa['Racist'])

#superseded
sns.boxplot(total_values)
sns.boxplot(r_values)
sns.boxplot(as_values)
sns.boxplot(d_values, color='teal')
sns.boxplot(f_values)
sns.boxplot(i_values)
sns.boxplot(h_values)
sns.boxplot(i_values)
sns.boxplot(t_values)


#or to look at all box plot distribution at once. 
ax = sns.boxplot(data=finalhatedataset, orient='h')

#middle line is median 
#the box is q1 to q3. first edge q1 end edge is q3 
#left whisker is minimum quartile . maximum quartile. 
#dots outside are outliers. 

finalhatedataset.sort_values(by=['Total All Incidents'], ascending=False).head(20)
finalhatedataset.sum()

#as there seems to be lots of outliers we can test for this. compute zscores

from scipy import stats

zscore = np.abs(stats.zscore(totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa))
print(zscore)
#set threshold for outliers.
threshold = 3

#print only the outliers
print(np.where(zscore>threshold))
#first array is a list of row numbers.  second array are the column numbers. 
#so you can slice on those specific values

print(zscore[140,7])


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
a = outliertruefalse.reset_index()
a.to_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\outliers_A.csv')

a.columns

# Melt DataFrame TO MAKE SUITABLE for seaborn
melted_outlierstruefalse = pd.melt(a, id_vars=["LSOA11CD"],var_name="OutlierTrueFalse") # first vars are variables to keep then Name of melted variable
melted_outlierstruefalse.head(30)

melted_outlierstruefalse.shape
outliertruefalse.shape

melted_outlierstruefalse

outlierscsv = melted_outlierstruefalse.groupby(['OutlierTrueFalse','value']).count()
outlierscsv
outlierscsv.to_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\outliers.csv')

#check there is a row for an LSOA for each hate crime type
melted_outlierstruefalse.loc[melted_outlierstruefalse['LSOA11CD'] =='E01000002']

#count the number of trues and false (outliers and non outliers)) for each hate crime
sns.set_style('whitegrid')
ax = sns.countplot(x="value", hue='OutlierTrueFalse', data=melted_outlierstruefalse, palette='cubehelix')
ax.set(xlabel='Outliers present', ylabel='Count')
plt.legend(title='Hate Crime Type', loc='upper right', fontsize=10, title_fontsize=12)

melted_outlierstruefalse

ax = sns.barplot(x="x", y="value", data=melted_outlierstruefalse, estimator=lambda x: len(x) / len(melted_outlierstruefalse) * 100)
ax.set(ylabel="Percent")


#LOG TRANSFORMATION. DONT THINK THIS IS THE WAY TO GO BECAUSE DATA IS COUNT DATA FOR HATE CRIMES.
inputarray = totalhatecrimesLSOA_by_type_clean_noNAN_nomissinglsoa.copy()
inputarray

output_array = np.log2(inputarray)
output_array
output_array.columns

sns.boxplot(output_array['Racist'])
sns.distplot(output_array['Total All Incidents'])
sns.distplot(output_array['Faith'])

sns.distplot(output_array)
output_array.columns

ax = sns.boxplot(data=output_array)


#transform back to original values
orig_values = (2**output_array)



















#%%



#a slim version of data frame with no dates. 
#df_slim = df[['LSOA','Grand Total','Type']]
#df_slim.set_index('LSOA').loc['E01000006']
#df_slim

#GET SPATIAL

import geopandas as gpd

lsoa = gpd.read_file('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/statistical-gis-boundaries-london/ESRI/LSOA_WGS84.shp')
lsoa = gpd.GeoDataFrame(lsoa,geometry='geometry',crs={'init':'epsg:4326'})
lsoa.crs


lsoa.crs
totalhatecrimesLSOA_by_type_clean_nonNAN.columns
totalhatecrimesLSOA_by_type_clean_nonNAN
totalhatecrimesLSOA_by_type_clean_nonNAN.reset_index()
lsoa_with_hatecrimes = pd.merge(lsoa,totalhatecrimesLSOA_by_type_clean_nonNAN, how='outer', left_on='LSOA11CD', right_on='LSOA')

lsoa_with_hatecrimes.columns

lsoa_with_hatecrimes.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_130720.shp")

#check original index is stil LSOA
totalhatecrimesLSOA_by_type_clean.index
totalhatecrimesLSOA_by_type_clean

#%%

#read in the entire excel workbook with multple sheet for LSOA additional info 
lsoa_data_1 = pd.read_excel('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/lsoa-data.xls', header=[0, 1], sheet_name='iadatasheet1')
lsoa_data_2 = pd.read_excel('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/lsoa-data.xls', header=[0, 1], sheet_name='iadatasheet2')

lsoa_data_1.columns
lsoa_data_2.columns

lsoa_data_1.index
lsoa_data_2.index

#check the index are matching. 
lsoa_data_1.loc[1111]
lsoa_data_2.loc[1111]

lsoa_data_1.loc[1234]
lsoa_data_2.loc[1234]


lsoa_data_1.loc[4321]
lsoa_data_2.loc[4321]

#index is matching

#merge on index
lsoa_data = pd.merge(lsoa_data_1,lsoa_data_2,left_index=True, right_index=True, how='outer')
lsoa_data.columns

lsoa_data.columns

lsoa_data.head(15)

#index still matches index number from original two data frames
lsoa_data.index

lsoa_data_simple = lsoa_data.copy()
lsoa_data_simple.shape
lsoa_data_simple.columns

#flatten the index to make it easier to work with
lsoa_data_simple.columns= lsoa_data_simple.columns.to_flat_index()
lsoa_data_simple.columns

lsoa_data_simple.head(15)

#access by tuple
lsoa_data_simple[('Codes_x', 'LSOACodes')]

lsoa_data_simple

lsoa_data_simple.rename(columns={('Codes_x', 'LSOACodes'): "LSOA_Codes"}, inplace=True)
lsoa_data_simple

#inspect all columns
list(lsoa_data_simple)

lsoa_data_simple.to_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\lsoa_datasimple.csv')
lsoa_data_simple.shape

#%% add in population for 2018

lsoa_2018pop = pd.read_excel(r'C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/LSOA 2018pop estimates/SAPE21DT1a-mid-2018-on-2019-LA-lsoa-syoa-estimates-formatted.xlsx', sheet_name='Mid-2018 Persons', skiprows=3)

lsoa_2018pop.head()

#rename first row to be column headers and drop the first from the data 
lsoa_2018pop.rename(columns=lsoa_2018pop.iloc[0], inplace = True)
lsoa_2018pop.drop([0], inplace = True)

lsoa_2018pop.head()

list(lsoa_2018pop)

lsoa_2018pop_allage = lsoa_2018pop[['Area Codes', 'LSOA','LA (2019 boundaries)', 'All Ages']]

lsoa_2018pop_allage

    


lsoa_all_data = pd.merge(lsoa_data_simple,lsoa_2018pop_allage, left_on='LSOA_Codes',right_on='Area Codes')


lsoa_all_data.sort_values(by=['All Ages'], ascending=False).head(20)

list(lsoa_all_data)

lsoa_all_data.columns = [str(s) for s in lsoa_all_data.columns]

list(lsoa_all_data)

type(lsoa)

type(lsoa_all_data)

lsoa_all_data.to_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\lsoa_all_data.csv')


#TEMPORARY TO NOT HAVE TO RUN WHOLE FILE
lsoa_all_data = pd.read_csv(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\lsoa_all_data.csv')
lsoa_with_hatecrimes=gpd.read_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_130720.shp")

list(lsoa_with_hatecrimes)
list(lsoa_all_data)


#merge the existing LSOA and hate crime gdf with all other LSOA data. 
lsoa_with_hatecrimes.columns
lsoa_all_data.columns



lsoa_with_hatecrimes_and_all_data = pd.merge(lsoa_with_hatecrimes,lsoa_all_data, how='outer', left_on='LSOA11CD', right_on='LSOA_Codes')


lsoa_all_data.sort_values(by=['All Ages'], ascending=False).head(20)

lsoa_with_hatecrimes_and_all_data.crs
type(lsoa_with_hatecrimes_and_all_data)


#calculate crime rate per 1000
list(lsoa_with_hatecrimes_and_all_data)
lsoa_with_hatecrimes_and_all_data['pop_per_1000'] = lsoa_with_hatecrimes_and_all_data['All Ages']/1000

lsoa_with_hatecrimes_and_all_data['All_HC_rate_p1000'] =lsoa_with_hatecrimes_and_all_data['Total All'] / lsoa_with_hatecrimes_and_all_data['pop_per_1000']


lsoa_with_hatecrimes_and_all_data.sort_values(by=['All_HC_rate_p1000'], ascending=False).head(20)

#remove all white space 
lsoa_with_hatecrimes_and_all_data.columns = lsoa_with_hatecrimes_and_all_data.columns.str.replace(' ', '_')

list(lsoa_with_hatecrimes_and_all_data)
type(lsoa_with_hatecrimes_and_all_data)


lsoa_with_hatecrimes_and_all_data.crs
lsoa_with_hatecrimes_and_all_data.to_csv(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_final.csv")
lsoa_with_hatecrimes_and_all_data.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_final.shp")
lsoa_with_hatecrimes_and_all_data.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_final.geojson", driver='GeoJSON')

#check original index is stil LSOA all data variable
lsoa_all_data.index


lsoa_with_hatecrimes_and_all_data.sort_values(by=['All_HC_rate_p1000'], ascending=False).head(20)
lsoa_with_hatecrimes_and_all_data.sort_values(by=['All_Ages'], ascending=False).head(20)


lsoa_with_hatecrimes_and_all_data.crs


#remove some columns for work in Arc
lsoa_with_hatecrimes_and_all_data.shape
list(lsoa_with_hatecrimes_and_all_data)
#create data frame to examine column position of columns to drop (to provide easy to see index number)
alist = list(lsoa_with_hatecrimes_and_all_data)
alist = pd.DataFrame(alist)
alist

cols_to_drop = np.r_[2,8:13, 28:306]
lsoa_with_hatecrimes_and_all_data_slim = lsoa_with_hatecrimes_and_all_data.drop(lsoa_with_hatecrimes_and_all_data.columns[cols_to_drop],axis=1)



lsoa_with_hatecrimes_and_all_data_slim.shape
list(lsoa_with_hatecrimes_and_all_data_slim)


lsoa_with_hatecrimes_and_all_data_slim.to_csv(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_SLIM_final_13072020.csv")
lsoa_with_hatecrimes_and_all_data_slim.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_SLIM_final_13072020.shp")
lsoa_with_hatecrimes_and_all_data_slim.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\lsoa_with_hatecrime_all_data_SLIM_final_13072020.geojson", driver='GeoJSON')


lsoa_with_hatecrimes_and_all_data_slim.info()


#%% GLOBAL MORANS I TEST 

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pysal as ps
from pysal import esda, weights
from esda.moran import Moran

import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster


                                                
lsoa_with_hatecrimes_and_all_data_slim.sort_values(by=['All_HC_rate_p1000'], ascending=False).head(20)

list(lsoa_with_hatecrimes_and_all_data_slim)


lsoa_with_hatecrimes_and_all_data_slim.plot()

lsoa4queen = gpd.read_file('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/statistical-gis-boundaries-london/ESRI/LSOA_WGS84.shp')
lsoa4queen.plot()

list(lsoa4queen)

lsoa4queen.dtypes

lsoa4queen.head()
lsoa4queen.set_index('LSOA11CD')

w = weights.Queen.from_dataframe(lsoa4queen, idVariable='LSOA11NM')
w.neighbors


qW = ps.queen_from_shapefile('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/statistical-gis-boundaries-london/ESRI/LSOA_WGS84.shp', idVariable='UniqueID')
qW.n
qW.neighbors[3]

W = pysal.weights.Queen.from_dataframe(lsoa4queen)

W.neighbors[8]

#create spatial weights matrix from lsoa shapefile. 
#w = ps.queen_from_shapefile('C:/Users/tho84231/OneDrive - University College London/Dissertation/Data/statistical-gis-boundaries-london/ESRI/LSOA_WGS84.shp', idVariable='LSOA11CD')
w

#then to see neighbours of an lsoa you can examine like :
w[4]


#w = lsoa_with_hatecrimes_and_all_data_slim.contiguityweights(graph=False)


#w.transform = "R"






















#%% MAAAAPPPPINNNNG

#join data to map and PLOT

lsoa.head()
lsoa.columns

lsoa.rename(columns={'LSOA11CD':'LSOA'}, inplace=True)

lsoa_withTotalincidents = lsoa.merge(totalhatecrimesLSOA,how='outer', on='LSOA')

lsoa_withTotalincidents.columns
#%%
from mpl_toolkits.axes_grid1 import make_axes_locatable

#this places the map in a figure. which is important. then you can turn off the axes.


 
f, ax = plt.subplots(1, figsize=(10,5))

#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="15%", pad=0.1)


ax = lsoa_withTotalincidents.plot(column = 'Total Incidents', cmap='magma',legend=True, ax=ax)
f.suptitle('Total hate crimes by LSOA')
#ax.set_axis_off()
ax.axis('off')
lims = plt.axis('equal')
plt.savefig('Total Incidents London.png',dpi=1080)#put save fig before show as show goes blank., 
plt.show()

lsoa_withTotalincidents.sort_values(by=['Total Incidents'], ascending=False).head(20)


lsoa_withTotalincidents.columns


print(type(lsoa)) 

print(type(lsoa_withTotalincidents)) 

#%% BOKEH AND JSON interactive?
#import json 


#convert the geodataframe to json for work with bokeh
#json_totalincidents = json.loads(lsoa_withTotalincidents.to_json())

#convert to string like object

#json_data = json.dumps(json_totalincidents)

#from bokeh.io import output_notebook, show, output_file
#from bokeh.plotting import figure
#from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
#from bokeh.palettes import brewer#Input GeoJSON source that contains features for plotting.
#geosource = GeoJSONDataSource(geojson = json_data)#Define a sequential multi-hue color palette.
#palette = brewer['YlGnBu'][8]#Reverse color order so that dark blue is highest obesity.
#palette = palette[::-1]#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
#color_mapper = LinearColorMapper(palette = palette, low = 0, high = 40)#Define custom tick labels for color bar.
#tick_labels = {'0': '0%', '5': '5%', '10':'10%', '15':'15%', '20':'20%', '25':'25%', '30':'30%','35':'35%', '40': '>40%'}#Create color bar. 
#color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
#border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)#Create figure object.
#p = figure(title = 'Share of adults who are obese, 2016', plot_height = 600 , plot_width = 950, toolbar_location = None)
#p.xgrid.grid_line_color = None
#p.ygrid.grid_line_color = None#Add patch renderer to figure. 
#p.patches('xs','ys', source = geosource,fill_color = {'field' :'per_cent_obesity', 'transform' : color_mapper},
          #line_color = 'black', line_width = 0.25, fill_alpha = 1)#Specify figure layout.
#p.add_layout(color_bar, 'below')#Display figure inline in Jupyter Notebook.
#output_notebook()#Display figure.
#show(p)

#%%
#plotly to make interactive maps. 
from plotly.offline import plot
import plotly.figure_factory as ff 
import plotly.graph_objs as go
import plotly.express as px
import plotly 

fig = go.Figure(data=[go.Bar(y=[1, 3, 2])])
plot(fig, auto_open=True)



data = px.data.gapminder()
px.data.gapminder().shape
px.data.gapminder().info()

data.tail()
value = px.data.gapminder().get('iso_alpha').value_counts()
value

sns.countplot(data=data, x='continent')

df1 = data.query("year==2007")



#fig = px.choropleth(df1, locations='iso_alpha', color= 'gdpPercap',hover_name='country', color_continuous_scale=px.colors.sequential.Purpor)
#fig.show()
#plot(fig, auto_open=True)

import json 

lsoa_withTotalincidents['id']=lsoa_withTotalincidents.index
#convert the geodataframe to json for work with PLOTLY
json_totalincidents = json.loads(lsoa_withTotalincidents.to_json())
#converts to string. 
#json_data = json.dumps(json_totalincidents, cls=plotly.utils.PlotlyJSONEncoder)


lsoa_withTotalincidents.columns

type(lsoa_withTotalincidents_geoJSON)
type(json_totalincidents)
type(lsoa_withTotalincidents)

lsoa_withTotalincidents.columns

lsoa_withTotalincidents_DF= pd.DataFrame(lsoa_withTotalincidents)

lsoa_withTotalincidents_DF.columns


fig1 = px.choropleth(lsoa_withTotalincidents_DF, geojson=json_totalincidents, locations=lsoa_withTotalincidents_DF['geometry'].astype(str), color= 'Total Incidents',hover_name='LSOA11NM', color_continuous_scale=px.colors.sequential.Purpor)
fig1.show()
plot(fig1, auto_open=True)








import plotly.express as px

df_final = px.data.election()
geojson = px.data.election_geojson()

type(geojson)
geojson['type']
geojson['features'][0].keys()

type(df_final)
type(lsoa_withTotalincidents_DF)

lsoa_withTotalincidents_DF.index

lsoa_withTotalincidents_DF['id']=lsoa_withTotalincidents_DF.index

lsoa_withTotalincidents_DF.columns

print(df_final["district"][2])
print(geojson["features"][0]["properties"])

print(geojson)

fig111 = px.choropleth(df_final, geojson=geojson, color="Bergeron",locations="district", featureidkey="properties.district",projection="mercator")
fig111.update_geos(fitbounds="locations", visible=False)
fig111.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig111.show()
plot(fig111, auto_open=True)


df_final.columns
geojson

type(json_totalincidents)
json_totalincidents['type']
json_totalincidents['features'][0].keys()

type(lsoa_withTotalincidents_DF)




print(lsoa_withTotalincidents_DF["LSOA"][2])
print(json_totalincidents["features"][0]["geometry"])

print(json_totalincidents)



json_totalincidents

figPLZ = px.choropleth(lsoa_withTotalincidents_DF['id'], geojson=json_totalincidents, color="Total Incidents",locations="id", featureidkey="properties.LSOA",projection="mercator")
#figPLZ.update_geos(fitbounds="locations", visible=False)
figPLZ.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#figPLZ.show()



figtry = go.Figure(go.Choroplethmapbox(geojson=json_totalincidents, locations=lsoa_withTotalincidents_DF.id, z=lsoa_withTotalincidents_DF.POPDEN,
                                    colorscale="Viridis"))
plot(figtry, auto_open=True)


lsoa_withTotalincidents_DF.columns


