# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:27:44 2020

@author: THO84231
"""

#%%

#import libraries 
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.formula.api as smf



#read in all the API results that had been saved
all_mosques_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_mosques_london_Gdf.shp')
all_mosques_gdf.crs

all_temples_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_temples_london_Gdf.shp')
all_temples_gdf.crs

all_churches_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_churches_london_Gdf.shp')
all_churches_gdf.crs

all_synagogues_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/all_synagogues_london_Gdf.shp')
all_synagogues_gdf.crs

all_community_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/community_london_Gdf.shp')
all_community_gdf.crs

all_disabled_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/disabled_london_Gdf.shp')
all_disabled_gdf.crs

all_wheelchair_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/wheelchairs_london_Gdf.shp')
all_wheelchair_gdf.crs

all_lgbt_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/LGBT_london_Gdf.shp')
all_lgbt_gdf.crs

all_trans_gdf = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/GoogleMapsAPI_results/transgender_london_Gdf.shp')
all_trans_gdf.crs


#rename the '0' column. 
all_mosques_gdf['Name'] = all_mosques_gdf['0']
all_mosques_gdf.head(20)
all_mosques_gdf.drop(['0'], axis=1, inplace=True)
all_mosques_gdf['Name']
all_mosques_gdf.columns
all_mosques_gdf.crs
all_mosques_gdf.plot()

all_temples_gdf['Name'] = all_temples_gdf['0']
all_temples_gdf.drop(['0'], axis=1, inplace=True)
all_temples_gdf['Name']
all_temples_gdf.crs
all_temples_gdf.plot()


all_churches_gdf['Name'] = all_churches_gdf['0']
all_churches_gdf.drop(['0'], axis=1, inplace=True)
all_churches_gdf['Name']
all_churches_gdf.crs
all_churches_gdf.plot()

all_synagogues_gdf['Name'] = all_synagogues_gdf['0']
all_synagogues_gdf.drop(['0'], axis=1, inplace=True)
all_synagogues_gdf['Name']
all_synagogues_gdf.crs
all_synagogues_gdf.plot()
list(all_synagogues_gdf)

all_community_gdf
all_community_gdf.crs
list(all_community_gdf)
all_community_gdf['Name'] = all_community_gdf['0']
all_community_gdf
all_community_gdf.drop(['0'], axis=1, inplace=True)
all_community_gdf
all_community_gdf['Name']
all_community_gdf.plot()
list(all_community_gdf)

all_disabled_gdf
all_disabled_gdf.crs
list(all_disabled_gdf)
all_disabled_gdf['Name'] = all_disabled_gdf['0']
all_disabled_gdf
all_disabled_gdf.drop(['0'], axis=1, inplace=True)
all_disabled_gdf
all_disabled_gdf['Name']
all_disabled_gdf.plot()
list(all_disabled_gdf)


all_wheelchair_gdf
all_wheelchair_gdf.crs
list(all_wheelchair_gdf)
all_wheelchair_gdf['Name'] = all_wheelchair_gdf['0']
all_wheelchair_gdf
all_wheelchair_gdf.drop(['0'], axis=1, inplace=True)
all_wheelchair_gdf
all_wheelchair_gdf['Name']
all_wheelchair_gdf.plot()
list(all_wheelchair_gdf)

all_lgbt_gdf
all_lgbt_gdf.crs
list(all_lgbt_gdf)
all_lgbt_gdf['Name'] = all_lgbt_gdf['0']
all_lgbt_gdf
all_lgbt_gdf.drop(['0'], axis=1, inplace=True)
all_lgbt_gdf
all_lgbt_gdf['Name']
all_lgbt_gdf.plot()
list(all_lgbt_gdf)

all_trans_gdf
all_trans_gdf.crs
list(all_trans_gdf)
all_trans_gdf['Name'] = all_trans_gdf['0']
all_trans_gdf
all_trans_gdf.drop(['0'], axis=1, inplace=True)
all_trans_gdf
all_trans_gdf['Name']
all_trans_gdf.plot()
list(all_trans_gdf)


#read in the LSOA and ALL data file 
finalhatedata = gpd.read_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/lsoa_with_hatecrime_and_population.geojson')
list(finalhatedata)
finalhatedata


finalhatedata.crs                  
finalhatedata.plot()

#work out how many features per LSOA
#first merge temples and churches into faith features. as well as disabled and wheelcahir into the disabled features 
#join the frames together into a list of frames
frames1 = [all_temples_gdf, all_churches_gdf]
#concat the 2 frames together
faith_places  = pd.concat(frames1)

#create new data frame of features for disabled 
frames2 = [all_wheelchair_gdf, all_disabled_gdf] 
disabled_places  = pd.concat(frames2)

#merge all different features into one data frame as well. 
frames3 = [all_wheelchair_gdf, all_disabled_gdf,all_temples_gdf,all_churches_gdf,all_mosques_gdf, all_synagogues_gdf,all_community_gdf,all_trans_gdf,all_lgbt_gdf] 
all_features  = pd.concat(frames3)




#check that the number of rows for each feature search separately is the same length as them merged together
all_temples_gdf.shape
all_churches_gdf.shape
faith_places.shape


all_wheelchair_gdf.shape
all_disabled_gdf.shape
disabled_places.shape


#this attaches the corresponding LSOA for every feature using a spatial join
list(finalhatedata)

mosques_in_LSOA = gpd.sjoin(all_mosques_gdf,finalhatedata)
synagogues_in_LSOA = gpd.sjoin(all_synagogues_gdf,finalhatedata)
disabled_places_in_LSOA = gpd.sjoin(disabled_places,finalhatedata)
faith_places_in_LSOA = gpd.sjoin(faith_places,finalhatedata)
all_community_in_LSOA = gpd.sjoin(all_community_gdf,finalhatedata)
all_trans_in_LSOA = gpd.sjoin(all_trans_gdf,finalhatedata)
all_lgbt_in_LSOA = gpd.sjoin(all_lgbt_gdf,finalhatedata)
all_features_in_LSOA = gpd.sjoin(all_features,finalhatedata)


#this then counts how many times each feature appears in each lsoa
count_of_mosques = mosques_in_LSOA.groupby('LSOA11CD').size()
#this checks that sum of all the features is still the same as features 
count_of_mosques.sum()

count_of_synagogues = synagogues_in_LSOA.groupby('LSOA11CD').size()
count_of_synagogues.sum()

count_of_disabled = disabled_places_in_LSOA.groupby('LSOA11CD').size()
count_of_disabled.sum()

count_of_faith = faith_places_in_LSOA.groupby('LSOA11CD').size()
count_of_faith.sum()

count_of_community = all_community_in_LSOA.groupby('LSOA11CD').size()
count_of_community.sum()


count_of_trans = all_trans_in_LSOA.groupby('LSOA11CD').size()
count_of_trans.sum()


count_of_lgbt = all_lgbt_in_LSOA.groupby('LSOA11CD').size()
count_of_lgbt.sum()

count_of_all_features = all_features_in_LSOA.groupby('LSOA11CD').size()
count_of_all_features.sum()

#join the count to the original lsoa data 
finalhatedata.columns
finalhatedata

finalhatedata.index
count_of_mosques.index
#set index as LSOA codes for the hate crime data so that the join with the count of features can join on the index 
finalhatedata.set_index('LSOA11CD', inplace=True)
finalhatedata.index
finalhatedata['mosque_count'] = count_of_mosques
finalhatedata['synagogues_count'] = count_of_synagogues
finalhatedata['disabledfeatures_count'] = count_of_disabled
finalhatedata['faithfeatures_count'] = count_of_faith
finalhatedata['community_count'] = count_of_community
finalhatedata['trans_count'] = count_of_trans
finalhatedata['lgbt_count'] = count_of_lgbt
finalhatedata['allfeatures_count'] = count_of_all_features

#check in QGIS that the exports of layers for hate crime LSOA and built environment match same counts produced by .loc below. 
finalhatedata.loc['E01000918']

#reset index to get the LSOA names back as column values. 
finalhatedata.reset_index(inplace=True)

finalhatedata

#export to file to save. 
#shapefile
finalhatedata.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final.shp')

#export to GEOJSON (for better column names) and CSV for use later
finalhatedata.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final.geojson', driver='GeoJSON')
finalhatedata.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final.csv')


#replace NAN for regression.
finalhatedata_nonNAN = finalhatedata.replace(np.nan,0)

#save outputs as geojson for GIS and csv for Regression
finalhatedata_nonNAN.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final_withoutNAN.geojson', driver='GeoJSON')
finalhatedata_nonNAN.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final_withoutNAN.csv')


#%%
#Observed vs Expected analysis 

finalhatedata_nonNAN = pd.read_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final_withoutNAN.csv')

list(finalhatedata_nonNAN)

#create observed - expected for anti semitic
finalhatedata_nonNAN['AntiSem_Ob-Exp'] = finalhatedata_nonNAN['Anti_Semitic'] - finalhatedata_nonNAN['synagogues_count'] * (finalhatedata_nonNAN['Anti_Semitic'].sum() / finalhatedata_nonNAN['synagogues_count'].sum())

#mosques and islamophobic
finalhatedata_nonNAN['Islamophobic_Ob-Exp'] = finalhatedata_nonNAN['Islamophobic'] - finalhatedata_nonNAN['mosque_count'] * (finalhatedata_nonNAN['Islamophobic'].sum() / finalhatedata_nonNAN['mosque_count'].sum())

#faith
finalhatedata_nonNAN['Faith_Ob-Exp'] = finalhatedata_nonNAN['Faith'] - finalhatedata_nonNAN['faithfeatures_count'] * (finalhatedata_nonNAN['Faith'].sum() / finalhatedata_nonNAN['faithfeatures_count'].sum())


#disabled 
finalhatedata_nonNAN['Disabled_Ob-Exp'] = finalhatedata_nonNAN['Disability'] - finalhatedata_nonNAN['disabledfeatures_count'] * (finalhatedata_nonNAN['Disability'].sum() / finalhatedata_nonNAN['disabledfeatures_count'].sum())

#Transgender 
finalhatedata_nonNAN['Transgender_Ob-Exp'] = finalhatedata_nonNAN['Transgender'] - finalhatedata_nonNAN['trans_count'] * (finalhatedata_nonNAN['Transgender'].sum() / finalhatedata_nonNAN['trans_count'].sum())


#homophobic and LGBT
finalhatedata_nonNAN['Homophobic_Ob-Exp'] = finalhatedata_nonNAN['Homophobic'] - finalhatedata_nonNAN['lgbt_count'] * (finalhatedata_nonNAN['Homophobic'].sum() / finalhatedata_nonNAN['lgbt_count'].sum())



#export and save for mapping
finalhatedata_nonNAN.to_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final_withoutNAN_withObExp.csv')





#%% Regression Analysis 
#Read back in CSV for regression
finalhatedata_nonNAN = pd.read_csv('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final_withoutNAN.csv')

#check column names
list(finalhatedata_nonNAN)
#check data fram
finalhatedata_nonNAN

#check distributions.
sns.distplot(finalhatedata_nonNAN['mosque_count'])
sns.distplot(finalhatedata_nonNAN['synagogues_count'])
sns.distplot(finalhatedata_nonNAN['disabledfeatures_count'])
sns.distplot(finalhatedata_nonNAN['faithfeatures_count'])
sns.distplot(finalhatedata_nonNAN['community_count'])
sns.distplot(finalhatedata_nonNAN['trans_count'])
sns.distplot(finalhatedata_nonNAN['lgbt_count'])
sns.distplot(finalhatedata_nonNAN['allfeatures_count'])

#Original data is all non normal distribution. Looks like poisson distribution. 

#EXAMINE THE INITIAL SCATTER PLOT between the hate crime rate and the count of associated buildings. 
#Anti-Semitic and Synagogues
ax = sns.scatterplot(x="synagogues_count", y="AntiSem_p1000", data=finalhatedata_nonNAN)
plt.title('Synagogues vs Anti-Semitic hate crimes')
# Set x-axis label
plt.xlabel('Count of synagogues')
# Set y-axis label
plt.ylabel('Anti-Semitic Hatecrime rate per 1000')
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/AntiSem_Syn.png")

#Islamophobic and Mosques
ax = sns.scatterplot(x="mosque_count", y="Islamophobic_p1000", data=finalhatedata_nonNAN)
plt.title('Mosques vs Islamaphobic hate crimes')
# Set x-axis label
plt.xlabel('Count of mosques')
# Set y-axis label
plt.ylabel('Islamophobic Hatecrime rate per 1000')
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Mosques_Islam.png")

#Disabled hate crime and disabled places
ax = sns.scatterplot(x="disabledfeatures_count", y="Disability_p1000", data=finalhatedata_nonNAN)
plt.title('Disabled friendly places vs Disabled hate crimes')
# Set x-axis label
plt.xlabel('Count of disabled friendly places')
# Set y-axis label
plt.ylabel('Disabled Hatecrime rate per 1000')
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Disab_Disabrate.png")

#Faith hate crime and faith associated buildings 
ax = sns.scatterplot(x="faithfeatures_count", y="Faith_p1000", data=finalhatedata_nonNAN)
plt.title('Faith places of worship vs Faith hate crimes')
# Set x-axis label
plt.xlabel('Count of places of worship')
# Set y-axis label
plt.ylabel('Faith Hatecrime rate per 1000')
ax.set_xticks(range(0,20,2)) # <--- set the ticks first
#ax.set_xticklabels(['0','2','4','6','2015','2016','2017','2018'])
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Faithplaces_vsfaithhatecrime.png")


#Transgender hate crimes and transgender associated buildings
ax = sns.scatterplot(x="trans_count", y="Transgender_p1000", data=finalhatedata_nonNAN)
plt.title('Transgender friendly places vs Transgender hate crimes')
# Set x-axis label
plt.xlabel('Count of transgender friendly places')
# Set y-axis label
plt.ylabel('Transgender Hatecrime rate per 1000')
ax.set_xticks(range(0,6,2))
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Trans_vs_transplaces.png")


#Homophobic hate crimes and LGBT associated buildings
ax = sns.scatterplot(x="lgbt_count", y="Homophobic_p1000", data=finalhatedata_nonNAN)
plt.title('Homosexual friendly places vs Homophobic hate crimes')
# Set x-axis label
plt.xlabel('Count of homosexual friendly places')
# Set y-axis label
plt.ylabel('Homophobic Hatecrime rate per 1000')
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Homo.png")

#Total hate crimes and all associated buildings and 
ax = sns.scatterplot(x="allfeatures_count", y="All_HC_rate_p1000", data=finalhatedata_nonNAN)
plt.title('All incidcators of community cohesion vs All hate crimes')
# Set x-axis label
plt.xlabel('Count of community cohesion indicators')
# Set y-axis label
plt.ylabel('Total Hatecrime rate per 1000')
#ax.set_xticks(range(0,6,2))
ax.figure.savefig("C:/Users/tho84231/Documents/GitHub/Dissertation/ScatterPlots/Allhatecrimes_and_allfeatures.png")


#%% poisson reg - Synagogues VS anti sem RATE

#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['AntiSem_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['AntiSem_p1000'].mean()))

#create train and test data frames. 
mask_antisem = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_antisem = finalhatedata_nonNAN[mask_antisem]
test_antisem = finalhatedata_nonNAN[~mask_antisem]

len(test_antisem)
len(train_antisem)
print('Training data set length='+str(len(train_antisem)))
print('Testing data set length='+str(len(test_antisem)))

#to list out column names
list(finalhatedata_nonNAN)

#set up the regression expression 
expr_antisem = """AntiSem_p1000 ~ synagogues_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_antisem, X_train_antisem = dmatrices(expr_antisem, train_antisem, return_type='dataframe')
y_test_antisem, X_test_antisem = dmatrices(expr_antisem, test_antisem, return_type='dataframe')

#train the model using the training matrices and run the poisson model. 
poisson_training_results_antisem = sm.GLM(y_train_antisem, X_train_antisem, family=sm.families.Poisson()).fit()
print(poisson_training_results_antisem.summary())


#%% NEGATIVE BINOMIAL - Synagogues VS ANTI SEMETIC HATE CRIME

##extract the vector of fitted rates calculated during the initial poisson regression model.
print(poisson_training_results_antisem.mu)
print(len(poisson_training_results_antisem.mu))

#Add the vector as a new column to the Data Frame of the training data set.
train_antisem['antisem_LAMBDA'] = poisson_training_results_antisem.mu

#create a new column to the values of the dependent variable of the OLS regression. This is to tell the model that this variable that is explained by the lambda value previously calculated
train_antisem['AUX_OLS_DEP_antisem'] = train_antisem.apply(lambda x: ((x['AntiSem_p1000'] - x['antisem_LAMBDA'])**2 - x['AntiSem_p1000']) / x['antisem_LAMBDA'], axis=1)

#set up the regression expression as before using the new columns. 
NB_expr_antisem = """AUX_OLS_DEP_antisem ~ antisem_LAMBDA - 1"""

#run the regression model and fit to the data. then extract value for lambda 
NB_olsr_results_antisem = smf.ols(NB_expr_antisem, train_antisem).fit()
print(NB_olsr_results_antisem.params)

#Call the t-score for the lambda value as below: 
NB_olsr_results_antisem.tvalues

#is significant, so can use train the NB model and fit this to the data
nb2_training_results_antisem = sm.GLM(y_train_antisem, X_train_antisem,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_antisem.params[0])).fit()
print(nb2_training_results_antisem.summary())





#%% poisson reg - MOSQUES PLACE VS ISLAMOPHOBIA RATE

#get columns 
list(finalhatedata_nonNAN)


#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['Islamophobic_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['Islamophobic_p1000'].mean()))

#create train and test data frames. 
mask_islam = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_islam = finalhatedata_nonNAN[mask_islam]
test_islam = finalhatedata_nonNAN[~mask_islam]

len(test_islam)
len(train_islam)
print('Training data set length='+str(len(train_islam)))
print('Testing data set length='+str(len(test_islam)))


#set up the regression expression 
expr_islam = """Islamophobic_p1000 ~ mosque_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_islam, X_train_islam = dmatrices(expr_islam, train_islam, return_type='dataframe')
y_test_islam, X_test_islam = dmatrices(expr_islam, test_islam, return_type='dataframe')

#train the model
poisson_training_results_islam = sm.GLM(y_train_islam, X_train_islam, family=sm.families.Poisson()).fit()
print(poisson_training_results_islam.summary())


#%% NEGATIVE BINOMIAL - MOSQUES VS ISLAMOPHOBIC HATE CRIME

##extract the vector of fitted rates calculated during the initial poisson regression model.

print(poisson_training_results_islam.mu)
print(len(poisson_training_results_islam.mu))

#Add the vector as a new column to the Data Frame of the training data set.
train_islam['islam_LAMBDA'] = poisson_training_results_islam.mu

#create a new column to the values of the dependent variable of the OLS regression. This is to tell the model that this variable that is explained by the lambda value previously calculated
train_islam['AUX_OLS_DEP_islam'] = train_islam.apply(lambda x: ((x['Islamophobic_p1000'] - x['islam_LAMBDA'])**2 - x['Islamophobic_p1000']) / x['islam_LAMBDA'], axis=1)
#set up regression expression
NB_expr_islam = """AUX_OLS_DEP_islam ~ islam_LAMBDA - 1"""

#fit the model to the data. 
NB_olsr_results_islam = smf.ols(NB_expr_islam, train_islam).fit()
#call the lambda result. 
print(NB_olsr_results_islam.params)
#check the associated tscore for the lambda value 
NB_olsr_results_islam.tvalues

#significant t score, means run the NB model. 
nb2_training_results_islam = sm.GLM(y_train_islam, X_train_islam,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_islam.params[0])).fit()
print(nb2_training_results_islam.summary())


#%% poisson reg - Disabled PLACE VS Disabled RATE

#columns
list(finalhatedata_nonNAN)

#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['Disability_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['Disability_p1000'].mean()))

#create train and test data frames. 
mask_disab = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_disab = finalhatedata_nonNAN[mask_disab]
test_disab = finalhatedata_nonNAN[~mask_disab]

len(test_disab)
len(train_disab)
print('Training data set length='+str(len(train_disab)))
print('Testing data set length='+str(len(test_disab)))


#set up the regression expression 
expr_disab = """Disability_p1000 ~ disabledfeatures_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_disab, X_train_disab = dmatrices(expr_disab, train_disab, return_type='dataframe')
y_test_disab, X_test_disab = dmatrices(expr_disab, test_disab, return_type='dataframe')

#train the model
poisson_training_results_disab = sm.GLM(y_train_disab, X_train_disab, family=sm.families.Poisson()).fit()
print(poisson_training_results_disab.summary())


#%% ZIP Model FOR DISABLED

#check zeros. 
ax = sns.distplot(finalhatedata_nonNAN['Disability_p1000'])
plt.title('Distribution of Disabled hate crime rate')

#run zip model on the data and print summary. 
zip_training_results_disab = sm.ZeroInflatedPoisson(endog=y_train_disab, exog=X_train_disab, exog_infl=X_train_disab, inflation='logit').fit()
print(zip_training_results_disab.summary())


#%% poisson reg - trans PLACE VS trans RATE

#columns
list(finalhatedata_nonNAN)

#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['Transgender_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['Transgender_p1000'].mean()))

#create train and test data frames. 
mask_trans = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_trans = finalhatedata_nonNAN[mask_trans]
test_trans = finalhatedata_nonNAN[~mask_trans]

len(test_trans)
len(train_trans)
print('Training data set length='+str(len(train_trans)))
print('Testing data set length='+str(len(test_trans)))

#set up the regression expression 
expr_trans = """Transgender_p1000 ~ trans_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_trans, X_train_trans = dmatrices(expr_trans, train_trans, return_type='dataframe')
y_test_trans, X_test_trans = dmatrices(expr_trans, test_trans, return_type='dataframe')

#train the model
poisson_training_results_trans = sm.GLM(y_train_trans, X_train_trans, family=sm.families.Poisson()).fit()
print(poisson_training_results_trans.summary())

#%% NEGATIVE BINOMIAL - trans VS TRANSGENDER HATE CRIME

#extract the vector of fitted rates calculated during the initial poisson regression model.
print(poisson_training_results_trans.mu)
print(len(poisson_training_results_trans.mu))

#store this vector in a new column
train_trans['trans_LAMBDA'] = poisson_training_results_trans.mu

#create a new column to the values of the dependent variable of the OLS regression. This is to tell the model that this variable that is explained by the lambda value previously calculated
train_trans['AUX_OLS_DEP_trans'] = train_trans.apply(lambda x: ((x['Transgender_p1000'] - x['trans_LAMBDA'])**2 - x['Transgender_p1000']) / x['trans_LAMBDA'], axis=1)

#create regression expression using new columns. 
NB_expr_trans = """AUX_OLS_DEP_trans ~ trans_LAMBDA - 1"""

#run the regression model and fit to the data. then extract value for lambda
NB_olsr_results_trans = smf.ols(NB_expr_trans, train_trans).fit()
print(NB_olsr_results_trans.params)
#check the tscore of lambda value
NB_olsr_results_trans.tvalues

#check significant and run the NB model
#run nb model and fit to the data. then print the summary. 
nb2_training_results_trans = sm.GLM(y_train_trans, X_train_trans,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_trans.params[0])).fit()
print(nb2_training_results_trans.summary())



#%% poisson reg - LGBT PLACE VS homophobic RATE

#get columns
list(finalhatedata_nonNAN)


#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['Homophobic_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['Homophobic_p1000'].mean()))

#create train and test data frames. 
mask_homophobic = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_homophobic = finalhatedata_nonNAN[mask_homophobic]
test_homophobic = finalhatedata_nonNAN[~mask_homophobic]

len(test_homophobic)
len(train_homophobic)
print('Training data set length='+str(len(train_homophobic)))
print('Testing data set length='+str(len(test_homophobic)))

#set up the regression expression 
expr_homophobic = """Homophobic_p1000 ~ lgbt_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_homophobic, X_train_homophobic = dmatrices(expr_homophobic, train_homophobic, return_type='dataframe')
y_test_homophobic, X_test_homophobic = dmatrices(expr_homophobic, test_homophobic, return_type='dataframe')

#train the model
poisson_training_results_homophobic = sm.GLM(y_train_homophobic, X_train_homophobic, family=sm.families.Poisson()).fit()
print(poisson_training_results_homophobic.summary())




#%% NEGATIVE BINOMIAL - LGBT VS homophobic HATE CRIME

#extract the vector of fitted rates calculated during the initial poisson regression model.
print(poisson_training_results_homophobic.mu)
print(len(poisson_training_results_homophobic.mu))

#Add the vector as a new column to the Data Frame of the training data set.
train_homophobic['homophobic_LAMBDA'] = poisson_training_results_homophobic.mu

#create a new column to the values of the dependent variable of the OLS regression. This is to tell the model that this variable that is explained by the lambda value previously calculated
train_homophobic['AUX_OLS_DEP_homophobic'] = train_homophobic.apply(lambda x: ((x['Homophobic_p1000'] - x['homophobic_LAMBDA'])**2 - x['Homophobic_p1000']) / x['homophobic_LAMBDA'], axis=1)

NB_expr_homophobic = """AUX_OLS_DEP_homophobic ~ homophobic_LAMBDA - 1"""

#run the regression model and fit to the data. then extract value for lambda
NB_olsr_results_homophobic = smf.ols(NB_expr_homophobic, train_homophobic).fit()
print(NB_olsr_results_homophobic.params)
#check the tscore of lambda to see its significant
NB_olsr_results_homophobic.tvalues

#run NB model and fit to data.  
nb2_training_results_homophobic = sm.GLM(y_train_homophobic, X_train_homophobic,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_homophobic.params[0])).fit()
print(nb2_training_results_homophobic.summary())

#%% poisson reg - FAITH PLACE VS FAITH RATE

#get column names
list(finalhatedata_nonNAN)


#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['Faith_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['Faith_p1000'].mean()))

#create train and test data frames. 
mask_faith = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_faith = finalhatedata_nonNAN[mask_faith]
test_faith = finalhatedata_nonNAN[~mask_faith]

len(test_faith)
len(train_faith)
print('Training data set length='+str(len(train_faith)))
print('Testing data set length='+str(len(test_faith)))

list(finalhatedata_nonNAN)

#set up the regression expression 
expr_faith = """Faith_p1000 ~ faithfeatures_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_faith, X_train_faith = dmatrices(expr_faith, train_faith, return_type='dataframe')
y_test_faith, X_test_faith = dmatrices(expr_faith, test_faith, return_type='dataframe')

#train the model
poisson_training_results_faith = sm.GLM(y_train_faith, X_train_faith, family=sm.families.Poisson()).fit()
print(poisson_training_results_faith.summary())



#%%    ZIP MODEL FOR FAITH AND FAITH FEATURES

#check zeros
ax = sns.distplot(finalhatedata_nonNAN['Faith_p1000'])
plt.title('Distribution of Faith rate')


zip_training_results_faith = sm.ZeroInflatedPoisson(endog=y_train_faith, exog=X_train_faith, exog_infl=X_train_faith, inflation='logit').fit()

print(zip_training_results_faith.summary())



#%% poisson reg - all associated buildings  VS all hate crimes rate

#get column names
list(finalhatedata_nonNAN)

#check the mean and variance 
print('variance='+str(finalhatedata_nonNAN['All_HC_rate_p1000'].var()))
print('mean='+str(finalhatedata_nonNAN['All_HC_rate_p1000'].mean()))

#create train and test data frames. 
mask_all2 = np.random.rand(len(finalhatedata_nonNAN)) < 0.8
train_all2 = finalhatedata_nonNAN[mask_all2]
test_all2 = finalhatedata_nonNAN[~mask_all2]

len(test_all2)
len(train_all2)
print('Training data set length='+str(len(train_all2)))
print('Testing data set length='+str(len(test_all2)))

#set up the regression expression 
expr_all2 = """All_HC_rate_p1000 ~ mosque_count + synagogues_count + disabledfeatures_count + faithfeatures_count + community_count + trans_count +lgbt_count  """

#Set up the X and y matrices for the training and testing data sets.
y_train_all2, X_train_all2 = dmatrices(expr_all2, train_all2, return_type='dataframe')
y_test_all2, X_test_all2 = dmatrices(expr_all2, test_all2, return_type='dataframe')

#train the model
poisson_training_results_all2 = sm.GLM(y_train_all2, X_train_all2, family=sm.families.Poisson()).fit()
print(poisson_training_results_all2.summary())




#%% NEGATIVE BINOMIAL - ALL SEPARATELY VS TOTAL HATE CRIME

#extract the vector of fitted rates calculated during the initial poisson regression model.
print(poisson_training_results_all2.mu)
print(len(poisson_training_results_all2.mu))

#Add the vector as a new column to the Data Frame of the training data set.
train_all2['all2_LAMBDA'] = poisson_training_results_all2.mu

#create a new column to the values of the dependent variable of the OLS regression. This is to tell the model that this variable that is explained by the lambda value previously calculated
train_all2['AUX_OLS_DEP_all2'] = train_all2.apply(lambda x: ((x['All_HC_rate_p1000'] - x['all2_LAMBDA'])**2 - x['All_HC_rate_p1000']) / x['all2_LAMBDA'], axis=1)

#set up regression expression using the new columns produced
NB_expr_all2 = """AUX_OLS_DEP_all2 ~ all2_LAMBDA - 1"""

#run the regression model and fit the data. then print the lambda value
NB_olsr_results_all2 = smf.ols(NB_expr_all2, train_all2).fit()
print(NB_olsr_results_all2.params)

#check the tscore of lambda to see its significant
NB_olsr_results_all2.tvalues

#once confirmed significant, run the NB model and fit the data. 
    
nb2_training_results_all2 = sm.GLM(y_train_all2, X_train_all2,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_all2.params[0])).fit()
print(nb2_training_results_all2.summary())
