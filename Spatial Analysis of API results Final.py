# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:27:44 2020

@author: THO84231
"""


#%%
#mosques spatial analysis
import seaborn as sns
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np




#%%

#count up the number of mosques per LSOA
import geopandas as gpd
#import seaborn as sns
import pandas as pd
#import matplotlib.pyplot as plt
#import numpy as np


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

#export to GEOJSON (for better column names)
finalhatedata.to_file('C:/Users/tho84231/Documents/GitHub/Dissertation/Output_Layers/LSOA_hate_data_and_Features_final.geojson', driver='GeoJSON')




#%%
#PLOTTING
# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot the equal interval choropleth and add a legend
lsoa_with_hatecrimes_and_all_data_slim_bng.plot(column='mosque_count', legend=True, axes=ax, colormap='Reds', linewidth=0.3)
boroughs_WGS84.plot(color='black', linewidth=0.3, ax=ax, alpha=0.5)
# Remove the axes
ax.set_axis_off()
# Set the title
ax.set_title("Equal Interval of Mosques in LSOA's accross")
# Keep axes proportionate
plt.axis('equal')
# Draw map
plt.show()


#%% EXPLORATORY DATA ANALYSIS FOR API RESULTS 


#remove nan for regression 

finalhatedata_bng_nonNAN = finalhatedata_bng.replace(np.nan,0)
finalhatedata_bng_nonNAN.to_file(r"C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\LSOA_HATE_DATA_and_Features_no_NAN.geojson", driver='GeoJSON')

#temp read back in
finalhatedata_bng_nonNAN = gpd.read_file(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Spatial Analysis\LSOA_HATE_DATA_and_Features_no_NAN.geojson')


finalhatedata_bng_nonNAN

#%% SPATIAL ANALYSIS REGRESSION 
import geopandas as gpd
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#temp read in 

#remove nan for regression 



list(finalhatedata_bng_nonNAN)


#check distributions
sns.distplot(finalhatedata_bng_nonNAN['mosque_count'])
#sns.distplot(finalhatedata_bng_nonNAN['temples_count'])
#sns.distplot(finalhatedata_bng_nonNAN['churches_count'])
sns.distplot(finalhatedata_bng_nonNAN['synagogues_count'])

sns.distplot(finalhatedata_bng_nonNAN['disabledfeatures_count'])
sns.distplot(finalhatedata_bng_nonNAN['faithfeatures_count'])
sns.distplot(finalhatedata_bng_nonNAN['community_count'])
sns.distplot(finalhatedata_bng_nonNAN['trans_count'])
sns.distplot(finalhatedata_bng_nonNAN['lgbt_count'])
sns.distplot(finalhatedata_bng_nonNAN['allfeatures_count'])


#data is non normal distribution. looks like poisson. 

#EXAMINE THE INITIAL SCATTER PLOT
list(finalhatedata_bng_nonNAN)
ax = sns.scatterplot(x="synagogues_count", y="AntiSemRat", data=finalhatedata_bng_nonNAN)
plt.title('Synagogues vs Anti-Semitic hate crimes')
# Set x-axis label
plt.xlabel('Count of synagogues')
# Set y-axis label
plt.ylabel('Anti-Semitic Hatecrime rate per 1000')
ax.figure.savefig("AntiSem_Syn2.png")


ax = sns.scatterplot(x="mosque_count", y="IslamRate", data=finalhatedata_bng_nonNAN)
plt.title('Mosques vs Islamaphobic hate crimes')
# Set x-axis label
plt.xlabel('Count of mosques')
# Set y-axis label
plt.ylabel('Islamophobic Hatecrime rate per 1000')
ax.figure.savefig("Mosques_Islam.png")


test = finalhatedata_bng_nonNAN[['mosque_count', 'IslamRate']]

ax = sns.heatmap(test)




ax = sns.scatterplot(x="disabledfeatures_count", y="DisabRate", data=finalhatedata_bng_nonNAN)
plt.title('Disabled friendly places vs Disabled hate crimes')
# Set x-axis label
plt.xlabel('Count of disabled friendly places')
# Set y-axis label
plt.ylabel('Disabled Hatecrime rate per 1000')
ax.figure.savefig("Disab_Disabrate.png")


ax = sns.scatterplot(x="faithfeatures_count", y="FaithRate", data=finalhatedata_bng_nonNAN)
plt.title('Faith places of worship vs Faith hate crimes')
# Set x-axis label
plt.xlabel('Count of places of worship')
# Set y-axis label
plt.ylabel('Faith Hatecrime rate per 1000')
ax.set_xticks(range(0,20,2)) # <--- set the ticks first
#ax.set_xticklabels(['0','2','4','6','2015','2016','2017','2018'])
ax.figure.savefig("Faithplaces_vsfaithhatecrime.png")





ax = sns.scatterplot(x="trans_count", y="TransRate", data=finalhatedata_bng_nonNAN)
plt.title('Transgender friendly places vs Transgender hate crimes')
# Set x-axis label
plt.xlabel('Count of transgender friendly places')
# Set y-axis label
plt.ylabel('Transgender Hatecrime rate per 1000')
ax.set_xticks(range(0,6,2))
ax.figure.savefig("Trans.png")



ax = sns.scatterplot(x="lgbt_count", y="HomoRate", data=finalhatedata_bng_nonNAN)
plt.title('Homosexual friendly places vs Homophobic hate crimes')
# Set x-axis label
plt.xlabel('Count of homosexual friendly places')
# Set y-axis label
plt.ylabel('Homophobic Hatecrime rate per 1000')
#ax.set_xticks(range(0,6,2))
ax.figure.savefig("Homo.png")

ax = sns.scatterplot(x="allfeatures_count", y="TotalRate", data=finalhatedata_bng_nonNAN)
plt.title('All incidcators of community cohesion vs All hate crimes')
# Set x-axis label
plt.xlabel('Count of community cohesion indicators')
# Set y-axis label
plt.ylabel('Total Hatecrime rate per 1000')
#ax.set_xticks(range(0,6,2))
ax.figure.savefig("All.png")


#maybe INTERESTING. BOX PLOT FOR EACH COUNT OF FEATURE
sns.catplot(x="synagogues_count", y="AntiSemRat", data=finalhatedata_bng_nonNAN, kind="box")

#POISSON REGRESION MODEL FOR COUNT DATA 




finalhatedata_bng_nonNAN.describe()
#%% poisson reg - Synagogues VS anti sem RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['AntiSemRat'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['AntiSemRat'].mean()))

#create train and test data frames. 
mask_antisem = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_antisem = finalhatedata_bng_nonNAN[mask_antisem]
test_antisem = finalhatedata_bng_nonNAN[~mask_antisem]

len(test_antisem)
len(train_antisem)
print('Training data set length='+str(len(train_antisem)))
print('Testing data set length='+str(len(test_antisem)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_antisem = """AntiSemRat ~ synagogues_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_antisem, X_train_antisem = dmatrices(expr_antisem, train_antisem, return_type='dataframe')
y_test_antisem, X_test_antisem = dmatrices(expr_antisem, test_antisem, return_type='dataframe')

#train the model
poisson_training_results_antisem = sm.GLM(y_train_antisem, X_train_antisem, family=sm.families.Poisson()).fit()
print(poisson_training_results_antisem.summary())

poisson_predictions_antisem = poisson_training_results_antisem.get_prediction(X_test_antisem)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_antisem = poisson_predictions_antisem.summary_frame()
print(predictions_summary_frame_antisem)

#plot the predicted vs actual
predicted_counts_antisem=predictions_summary_frame_antisem['mean']
actual_counts_antisem = y_test_antisem['AntiSemRat']
fig = plt.figure()
fig.suptitle('Predicted versus actual Anti Semotoc hate crime incidents in London')
predicted, = plt.plot(X_test_antisem.index, predicted_counts_antisem, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_antisem.index, actual_counts_antisem, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for Anti Semitic hate crimes')
plt.scatter(x=predicted_counts_antisem, y=actual_counts_antisem, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')

#%% NEGATIVE BINOMIAL - Synagogues VS ANTI SEMETIC HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_antisem.mu)
print(len(poisson_training_results_antisem.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_antisem['antisem_LAMBDA'] = poisson_training_results_antisem.mu

train_antisem['AUX_OLS_DEP_antisem'] = train_antisem.apply(lambda x: ((x['AntiSemRat'] - x['antisem_LAMBDA'])**2 - x['AntiSemRat']) / x['antisem_LAMBDA'], axis=1)

NB_expr_antisem = """AUX_OLS_DEP_antisem ~ antisem_LAMBDA - 1"""


NB_olsr_results_antisem = smf.ols(NB_expr_antisem, train_antisem).fit()
print(NB_olsr_results_antisem.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '1.067987' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_antisem.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  3.777204
#We conclude that a=1.067987 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_antisem = sm.GLM(y_train_antisem, X_train_antisem,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_antisem.params[0])).fit()
print(nb2_training_results_antisem.summary())

#PREDICTIONS USING NB2

nb2_predictions_antisem = nb2_training_results_antisem.get_prediction(X_test_antisem)


nb2_predictions_summary_frame_antisem = nb2_predictions_antisem.summary_frame()
print(nb2_predictions_summary_frame_antisem)

#you can then plot these
nb2_predicted_counts_antisem=nb2_predictions_summary_frame_antisem['mean']
actual_counts_antisem = y_test_antisem['AntiSemRat']
nb2_df_antisem = pd.DataFrame({'Actual':actual_counts_antisem , 'Predicted':nb2_predicted_counts_antisem })
nb2_df_antisem

nb2_df_antisem.sort_values(['Predicted'], ascending=False)

#nb2_df_faith = nb2_df_faith.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_antisem)

#%% GENERALISED REG FOR synagogues  VS anti sem
#generalised poisson as the mean and variance are not the same. 
#Build Consul’s Generalized Poison regression model, know as GP-1 for all hate crimes and faith features
    

#check the mean and variance 

print('variance='+str(finalhatedata_bng_nonNAN['AntiSemRat'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['AntiSemRat'].mean()))

#the mean and variance are very different for total hate crime incidence. 

#use same mask, training and test data as for poisson . see above. 

#same regression expression as before for normal poisson. see above. 

gen_poisson_gp1_antisem = sm.GeneralizedPoisson(y_train_antisem, X_train_antisem, p=1)

#fit (train) the model
gen_poisson_gp1_results_antisem = gen_poisson_gp1_antisem.fit()
print(gen_poisson_gp1_results_antisem.summary())

#predict using the GP model. 
gen_poisson_gp1_predictions_antisem = gen_poisson_gp1_results_antisem.predict(X_test_antisem)
#gen_poisson_gp1_predictions is a pandas Series object that contains the predicted  count for 
#each row in the X_test matrix. Remember that y_test contains the actual observed counts.

#plot the predicted vs actual
gp_predicted_counts_antisem = gen_poisson_gp1_predictions_antisem
actual_counts_antisem = y_test_antisem['AntiSemRat']
fig = plt.figure()
fig.suptitle('Predicted versus actual hate crimes GP model')
predicted, = plt.plot(X_test_antisem.index, gp_predicted_counts_antisem, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_antisem.index, actual_counts_antisem, 'ro-', label='Actual counts', alpha=0.5)
plt.legend(handles=[predicted, actual])
plt.show()

#make data frame for easy plotting

gp1_df_antisem = pd.DataFrame({'Actual':actual_counts_antisem , 'Predicted':gp_predicted_counts_antisem })
gp1_df_antisem.sort_values(['Predicted'], ascending=False)


ax = plt.gca()
gp1_df_antisem.plot(kind='line',y='Actual',ax=ax)
gp1_df_antisem.plot(kind='line',y='Predicted', color='red', alpha=0.5, ax=ax)
plt.title('Anti-semitic Hate Crime predictions VS actual')

gp1_df_antisem.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


fig, ax = plt.subplots()
ax.scatter(actual_counts_antisem, gp_predicted_counts_antisem)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

ax = sns.scatterplot(x='Actual', y='Predicted', data= gp1_df_antisem)


#%% poisson reg - FAITH PLACE VS FAITH RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['FaithRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['FaithRate'].mean()))

#create train and test data frames. 
mask_faith = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_faith = finalhatedata_bng_nonNAN[mask_faith]
test_faith = finalhatedata_bng_nonNAN[~mask_faith]

len(test_faith)
len(train_faith)
print('Training data set length='+str(len(train_faith)))
print('Testing data set length='+str(len(test_faith)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_faith = """FaithRate ~ faithfeatures_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_faith, X_train_faith = dmatrices(expr_faith, train_faith, return_type='dataframe')
y_test_faith, X_test_faith = dmatrices(expr_faith, test_faith, return_type='dataframe')

#train the model
poisson_training_results_faith = sm.GLM(y_train_faith, X_train_faith, family=sm.families.Poisson()).fit()
print(poisson_training_results_faith.summary())

poisson_predictions_faith = poisson_training_results_faith.get_prediction(X_test_faith)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_faith = poisson_predictions_faith.summary_frame()
print(predictions_summary_frame_faith)

#plot the predicted vs actual
predicted_counts_faith=predictions_summary_frame_faith['mean']
actual_counts_faith = y_test_faith['FaithRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual Faith hate crime incidents in London')
predicted, = plt.plot(X_test_faith.index, predicted_counts_faith, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_faith.index, actual_counts_faith, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for Faith hate crimes')
plt.scatter(x=predicted_counts_faith, y=actual_counts_faith, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')

#%%    ZIP MODEL FOR FAITH AND FAITH FEATURES

import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#check zeros. LOTS OF ZEROS
ax = sns.distplot(finalhatedata_bng_nonNAN['FaithRate'])
plt.title('Distribution of Faith rate')
ax.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Graphs and Figures\FaithDist')

#Using statsmodels’s ZeroInflatedPoisson class, let’s build and train a ZIP regression model on the training data set.


zip_training_results_faith = sm.ZeroInflatedPoisson(endog=y_train_faith, exog=X_train_faith, exog_infl=X_train_faith, inflation='logit').fit()

print(zip_training_results_faith.summary())



#TRY PREDICTION

zip_predictions_faith = zip_training_results_faith.predict(X_test_faith,exog_infl=X_test_faith)
zip_predicted_counts_faith =np.round(zip_predictions_faith)
actual_counts_faith = y_test_faith['FaithRate']

print('ZIP RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(zip_predicted_counts_faith,actual_counts_faith),2)))))

#PLOT 
fig = plt.figure()

fig.suptitle('Predicted versus actual counts using the ZIP model')

predicted, = plt.plot(X_test_faith.index, zip_predicted_counts_faith, 'go-', label='Predicted')
actual, = plt.plot(X_test_faith.index, actual_counts_faith, 'ro-', label='Actual')
plt.legend(handles=[predicted, actual])
plt.show()





#%% SUPERSEDED NOT NEEDED
#NEGATIVE BINOMIAL - FAITH FEATURES VS FAITH HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_faith.mu)
print(len(poisson_training_results_faith.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_faith['Faith_LAMBDA'] = poisson_training_results_faith.mu

train_faith['AUX_OLS_DEP_faith'] = train_faith.apply(lambda x: ((x['FaithRate'] - x['Faith_LAMBDA'])**2 - x['FaithRate']) / x['Faith_LAMBDA'], axis=1)

NB_expr_faith = """AUX_OLS_DEP_faith ~ Faith_LAMBDA - 1"""


NB_olsr_results_faith = smf.ols(NB_expr_faith, train_faith).fit()
print(NB_olsr_results_faith.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '473.21802' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_faith.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  8.065505 
#We conclude that a=473.21802 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_faith = sm.GLM(y_train_faith, X_train_faith,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_faith.params[0])).fit()
print(nb2_training_results_faith.summary())

#PREDICTIONS USING NB2

nb2_predictions_faith = nb2_training_results_faith.get_prediction(X_test_faith)


nb2_predictions_summary_frame_faith = nb2_predictions_faith.summary_frame()
print(nb2_predictions_summary_frame_faith)

#you can then plot these
nb2_predicted_counts_faith=nb2_predictions_summary_frame_faith['mean']
actual_counts_faith = y_test_faith['FaithRate']
nb2_df_faith = pd.DataFrame({'Actual':actual_counts_faith , 'Predicted':nb2_predicted_counts_faith })
nb2_df_faith

nb2_df_faith.sort_values(['Predicted'], ascending=False)

#nb2_df_faith = nb2_df_faith.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_faith)



#%% poisson reg - MOSQUES PLACE VS ISLAMOPHOBIA RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['IslamRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['IslamRate'].mean()))

#create train and test data frames. 
mask_islam = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_islam = finalhatedata_bng_nonNAN[mask_islam]
test_islam = finalhatedata_bng_nonNAN[~mask_islam]

len(test_islam)
len(train_islam)
print('Training data set length='+str(len(train_islam)))
print('Testing data set length='+str(len(test_islam)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_islam = """IslamRate ~ mosque_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_islam, X_train_islam = dmatrices(expr_islam, train_islam, return_type='dataframe')
y_test_islam, X_test_islam = dmatrices(expr_islam, test_islam, return_type='dataframe')

#train the model
poisson_training_results_islam = sm.GLM(y_train_islam, X_train_islam, family=sm.families.Poisson()).fit()
print(poisson_training_results_islam.summary())

poisson_predictions_islam = poisson_training_results_islam.get_prediction(X_test_islam)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_islam = poisson_predictions_islam.summary_frame()
print(predictions_summary_frame_islam)

#plot the predicted vs actual
predicted_counts_islam=predictions_summary_frame_islam['mean']
actual_counts_islam = y_test_islam['IslamRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual Islamophobic hate crime incidents in London')
predicted, = plt.plot(X_test_islam.index, predicted_counts_islam, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_islam.index, actual_counts_islam, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for Islamophobic hate crimes')
plt.scatter(x=predicted_counts_islam, y=actual_counts_islam, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')


#%% NEGATIVE BINOMIAL - MOSQUES VS ISLAMOPHOBIC HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_islam.mu)
print(len(poisson_training_results_islam.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_islam['islam_LAMBDA'] = poisson_training_results_islam.mu

train_islam['AUX_OLS_DEP_islam'] = train_islam.apply(lambda x: ((x['IslamRate'] - x['islam_LAMBDA'])**2 - x['IslamRate']) / x['islam_LAMBDA'], axis=1)

NB_expr_islam = """AUX_OLS_DEP_islam ~ islam_LAMBDA - 1"""


NB_olsr_results_islam = smf.ols(NB_expr_islam, train_islam).fit()
print(NB_olsr_results_islam.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '473.21802' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_islam.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  8.065505 
#We conclude that a=473.21802 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_islam = sm.GLM(y_train_islam, X_train_islam,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_islam.params[0])).fit()
print(nb2_training_results_islam.summary())

#PREDICTIONS USING NB2

nb2_predictions_islam = nb2_training_results_islam.get_prediction(X_test_islam)


nb2_predictions_summary_frame_islam = nb2_predictions_islam.summary_frame()
print(nb2_predictions_summary_frame_islam)

#you can then plot these
nb2_predicted_counts_islam=nb2_predictions_summary_frame_islam['mean']
actual_counts_islam = y_test_islam['IslamRate']
nb2_df_islam = pd.DataFrame({'Actual':actual_counts_islam , 'Predicted':nb2_predicted_counts_islam })
nb2_df_islam

nb2_df_islam.sort_values(['Predicted'], ascending=False)

#nb2_df_islam = nb2_df_islam.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_islam)


#%% GENERALISED REG FOR mosques  VS islamophobic
#generalised poisson as the mean and variance are not the same. 
#Build Consul’s Generalized Poison regression model, know as GP-1 for all hate crimes and faith features
    

#check the mean and variance 

print('variance='+str(finalhatedata_bng_nonNAN['IslamRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['IslamRate'].mean()))

#the mean and variance are very different for total hate crime incidence. 

#use same mask, training and test data as for poisson . see above. 

#same regression expression as before for normal poisson. see above. 

gen_poisson_gp1_islam = sm.GeneralizedPoisson(y_train_islam, X_train_islam, p=1)

#fit (train) the model
gen_poisson_gp1_results_islam = gen_poisson_gp1_islam.fit()
print(gen_poisson_gp1_results_islam.summary())

#predict using the GP model. 
gen_poisson_gp1_predictions_islam = gen_poisson_gp1_results_islam.predict(X_test_islam)
#gen_poisson_gp1_predictions is a pandas Series object that contains the predicted  count for 
#each row in the X_test matrix. Remember that y_test contains the actual observed counts.

#plot the predicted vs actual
gp_predicted_counts_islam = gen_poisson_gp1_predictions_islam
actual_counts_islam = y_test_islam['IslamRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual hate crimes GP model')
predicted, = plt.plot(X_test_islam.index, gp_predicted_counts_islam, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_islam.index, actual_counts_islam, 'ro-', label='Actual counts', alpha=0.5)
plt.legend(handles=[predicted, actual])
plt.show()

#make data frame for easy plotting

gp1_df_islam = pd.DataFrame({'Actual':actual_counts_islam , 'Predicted':gp_predicted_counts_islam })
gp1_df_islam.sort_values(['Predicted'], ascending=False)


ax = plt.gca()
gp1_df_islam.plot(kind='line',y='Actual',ax=ax)
gp1_df_islam.plot(kind='line',y='Predicted', color='red', alpha=0.5, ax=ax)
plt.title('Islamophobic Hate Crime predictions VS actual')

gp1_df_islam.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


fig, ax = plt.subplots()
ax.scatter(actual_counts_islam, gp_predicted_counts_islam)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

ax = sns.scatterplot(x='Actual', y='Predicted', data= gp1_df_islam)


#%% poisson reg - Disabled PLACE VS Disabled RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['DisabRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['DisabRate'].mean()))

#create train and test data frames. 
mask_disab = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_disab = finalhatedata_bng_nonNAN[mask_disab]
test_disab = finalhatedata_bng_nonNAN[~mask_disab]

len(test_disab)
len(train_disab)
print('Training data set length='+str(len(train_disab)))
print('Testing data set length='+str(len(test_disab)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_disab = """DisabRate ~ disabledfeatures_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_disab, X_train_disab = dmatrices(expr_disab, train_disab, return_type='dataframe')
y_test_disab, X_test_disab = dmatrices(expr_disab, test_disab, return_type='dataframe')

#train the model
poisson_training_results_disab = sm.GLM(y_train_disab, X_train_disab, family=sm.families.Poisson()).fit()
print(poisson_training_results_disab.summary())

poisson_predictions_disab = poisson_training_results_disab.get_prediction(X_test_disab)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_disab = poisson_predictions_disab.summary_frame()
print(predictions_summary_frame_disab)

#plot the predicted vs actual
predicted_counts_disab=predictions_summary_frame_disab['mean']
actual_counts_disab = y_test_disab['DisabRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual Disabled hate crime incidents in London')
predicted, = plt.plot(X_test_disab.index, predicted_counts_disab, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_disab.index, actual_counts_disab, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for disabled hate crimes')
plt.scatter(x=predicted_counts_disab, y=actual_counts_disab, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')


#%% ZIP FOR DISABLED

import pandas as pd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

#check zeros. LOTS OF ZEROS
ax = sns.distplot(finalhatedata_bng_nonNAN['DisabRate'])
plt.title('Distribution of Disabled hate crime rate')
ax.figure.savefig(r'C:\Users\tho84231\OneDrive - University College London\Dissertation\Data\Graphs and Figures\DisabDist')

#Using statsmodels’s ZeroInflatedPoisson class, let’s build and train a ZIP regression model on the training data set.


zip_training_results_disab = sm.ZeroInflatedPoisson(endog=y_train_disab, exog=X_train_disab, exog_infl=X_train_disab, inflation='logit').fit()

print(zip_training_results_disab.summary())



#TRY PREDICTION

zip_predictions_disab = zip_training_results_disab.predict(X_test_disab,exog_infl=X_test_disab)
zip_predicted_counts_disab =np.round(zip_predictions_disab)
actual_counts_disab = y_test_disab['DisabRate']

print('ZIP RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(zip_predicted_counts_disab,actual_counts_disab),2)))))

#PLOT 
fig = plt.figure()

fig.suptitle('Predicted versus actual counts using the ZIP model')

predicted, = plt.plot(X_test_disab.index, zip_predicted_counts_disab, 'go-', label='Predicted')
actual, = plt.plot(X_test_disab.index, actual_counts_disab, 'ro-', label='Actual')
plt.legend(handles=[predicted, actual])
plt.show()




#%% poisson reg - trans PLACE VS trans RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['TransRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['TransRate'].mean()))

#create train and test data frames. 
mask_trans = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_trans = finalhatedata_bng_nonNAN[mask_trans]
test_trans = finalhatedata_bng_nonNAN[~mask_trans]

len(test_trans)
len(train_trans)
print('Training data set length='+str(len(train_trans)))
print('Testing data set length='+str(len(test_trans)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_trans = """TransRate ~ trans_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_trans, X_train_trans = dmatrices(expr_trans, train_trans, return_type='dataframe')
y_test_trans, X_test_trans = dmatrices(expr_trans, test_trans, return_type='dataframe')

#train the model
poisson_training_results_trans = sm.GLM(y_train_trans, X_train_trans, family=sm.families.Poisson()).fit()
print(poisson_training_results_trans.summary())

poisson_predictions_trans = poisson_training_results_trans.get_prediction(X_test_trans)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_trans = poisson_predictions_trans.summary_frame()
print(predictions_summary_frame_trans)

#plot the predicted vs actual
predicted_counts_trans=predictions_summary_frame_trans['mean']
actual_counts_trans = y_test_trans['TransRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual Transgender hate crime incidents in London')
predicted, = plt.plot(X_test_trans.index, predicted_counts_trans, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_trans.index, actual_counts_trans, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for transgender hate crimes')
plt.scatter(x=predicted_counts_trans, y=actual_counts_trans, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')


#%% NEGATIVE BINOMIAL - trans VS TRANSGENDER HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_trans.mu)
print(len(poisson_training_results_trans.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_trans['trans_LAMBDA'] = poisson_training_results_trans.mu

train_trans['AUX_OLS_DEP_trans'] = train_trans.apply(lambda x: ((x['TransRate'] - x['trans_LAMBDA'])**2 - x['TransRate']) / x['trans_LAMBDA'], axis=1)

NB_expr_trans = """AUX_OLS_DEP_trans ~ trans_LAMBDA - 1"""


NB_olsr_results_trans = smf.ols(NB_expr_trans, train_trans).fit()
print(NB_olsr_results_trans.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '3.316158' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_trans.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  2.802314
#We conclude that a=3.316158 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_trans = sm.GLM(y_train_trans, X_train_trans,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_trans.params[0])).fit()
print(nb2_training_results_trans.summary())

#PREDICTIONS USING NB2

nb2_predictions_trans = nb2_training_results_trans.get_prediction(X_test_trans)


nb2_predictions_summary_frame_trans = nb2_predictions_trans.summary_frame()
print(nb2_predictions_summary_frame_trans)

#you can then plot these
nb2_predicted_counts_trans=nb2_predictions_summary_frame_trans['mean']
actual_counts_trans = y_test_trans['TransRate']
nb2_df_trans = pd.DataFrame({'Actual':actual_counts_trans , 'Predicted':nb2_predicted_counts_trans })
nb2_df_trans

nb2_df_trans.sort_values(['Predicted'], ascending=False)

#nb2_df_trans = nb2_df_trans.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_trans)



#%% GENERALISED REG FOR trans  VS transrate 
#generalised poisson as the mean and variance are not the same. 
#Build Consul’s Generalized Poison regression model, know as GP-1 for all hate crimes and faith features
    

#check the mean and variance 

print('variance='+str(finalhatedata_bng_nonNAN['TransRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['TransRate'].mean()))

#the mean and variance are very different for total hate crime incidence. 

#use same mask, training and test data as for poisson . see above. 

#same regression expression as before for normal poisson. see above. 

gen_poisson_gp1_trans = sm.GeneralizedPoisson(y_train_trans, X_train_trans, p=1)

#fit (train) the model
gen_poisson_gp1_results_trans = gen_poisson_gp1_trans.fit()
print(gen_poisson_gp1_results_trans.summary())

#predict using the GP model. 
gen_poisson_gp1_predictions_trans = gen_poisson_gp1_results_trans.predict(X_test_trans)
#gen_poisson_gp1_predictions is a pandas Series object that contains the predicted  count for 
#each row in the X_test matrix. Remember that y_test contains the actual observed counts.

#plot the predicted vs actual
gp_predicted_counts_trans = gen_poisson_gp1_predictions_trans
actual_counts_trans = y_test_trans['TransRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual hate crimes GP model')
predicted, = plt.plot(X_test_trans.index, gp_predicted_counts_trans, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_trans.index, actual_counts_trans, 'ro-', label='Actual counts', alpha=0.5)
plt.legend(handles=[predicted, actual])
plt.show()

#make data frame for easy plotting

gp1_df_trans = pd.DataFrame({'Actual':actual_counts_trans , 'Predicted':gp_predicted_counts_trans })
gp1_df_trans.sort_values(['Predicted'], ascending=False)


ax = plt.gca()
gp1_df_trans.plot(kind='line',y='Actual',ax=ax)
gp1_df_trans.plot(kind='line',y='Predicted', color='red', alpha=0.5, ax=ax)
plt.title('Transgender Hate Crime predictions VS actual')

gp1_df_trans.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


fig, ax = plt.subplots()
ax.scatter(actual_counts_trans, gp_predicted_counts_trans)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

ax = sns.scatterplot(x='Actual', y='Predicted', data= gp1_df_trans)



#%% poisson reg - LGBT PLACE VS homophobic RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['HomoRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['HomoRate'].mean()))

#create train and test data frames. 
mask_homo = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_homo = finalhatedata_bng_nonNAN[mask_homo]
test_homo = finalhatedata_bng_nonNAN[~mask_homo]

len(test_homo)
len(train_homo)
print('Training data set length='+str(len(train_homo)))
print('Testing data set length='+str(len(test_homo)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_homo = """HomoRate ~ lgbt_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_homo, X_train_homo = dmatrices(expr_homo, train_homo, return_type='dataframe')
y_test_homo, X_test_homo = dmatrices(expr_homo, test_homo, return_type='dataframe')

#train the model
poisson_training_results_homo = sm.GLM(y_train_homo, X_train_homo, family=sm.families.Poisson()).fit()
print(poisson_training_results_homo.summary())

poisson_predictions_homo = poisson_training_results_homo.get_prediction(X_test_homo)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_homo = poisson_predictions_homo.summary_frame()
print(predictions_summary_frame_homo)

#plot the predicted vs actual
predicted_counts_homo=predictions_summary_frame_homo['mean']
actual_counts_homo = y_test_homo['HomoRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual Homophobic hate crime incidents in London')
predicted, = plt.plot(X_test_homo.index, predicted_counts_homo, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_homo.index, actual_counts_homo, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for homophobic hate crimes')
plt.scatter(x=predicted_counts_homo, y=actual_counts_homo, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')



#%% NEGATIVE BINOMIAL - LGBT VS homophobic HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_homo.mu)
print(len(poisson_training_results_homo.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_homo['homo_LAMBDA'] = poisson_training_results_homo.mu

train_homo['AUX_OLS_DEP_homo'] = train_homo.apply(lambda x: ((x['HomoRate'] - x['homo_LAMBDA'])**2 - x['HomoRate']) / x['homo_LAMBDA'], axis=1)

NB_expr_homo = """AUX_OLS_DEP_homo ~ homo_LAMBDA - 1"""


NB_olsr_results_homo = smf.ols(NB_expr_homo, train_homo).fit()
print(NB_olsr_results_homo.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '1.4988' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_homo.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  3.387404
#We conclude that a=1.4988 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_homo = sm.GLM(y_train_homo, X_train_homo,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_homo.params[0])).fit()
print(nb2_training_results_homo.summary())

#PREDICTIONS USING NB2

nb2_predictions_homo = nb2_training_results_homo.get_prediction(X_test_homo)


nb2_predictions_summary_frame_homo = nb2_predictions_homo.summary_frame()
print(nb2_predictions_summary_frame_homo)

#you can then plot these
nb2_predicted_counts_homo=nb2_predictions_summary_frame_homo['mean']
actual_counts_homo = y_test_homo['HomoRate']
nb2_df_homo = pd.DataFrame({'Actual':actual_counts_homo , 'Predicted':nb2_predicted_counts_homo })
nb2_df_homo

nb2_df_homo.sort_values(['Predicted'], ascending=False)

#nb2_df_homo = nb2_df_homo.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_homo)


#%% GENERALISED REG FOR LGBT  VS homophobic 
#generalised poisson as the mean and variance are not the same. 
#Build Consul’s Generalized Poison regression model, know as GP-1 for all hate crimes and faith features
    

#check the mean and variance 

print('variance='+str(finalhatedata_bng_nonNAN['HomoRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['HomoRate'].mean()))

#the mean and variance are very different for total hate crime incidence. 

#use same mask, training and test data as for poisson . see above. 

#same regression expression as before for normal poisson. see above. 

gen_poisson_gp1_homo = sm.GeneralizedPoisson(y_train_homo, X_train_homo, p=1)

#fit (train) the model
gen_poisson_gp1_results_homo = gen_poisson_gp1_homo.fit()
print(gen_poisson_gp1_results_homo.summary())

#predict using the GP model. 
gen_poisson_gp1_predictions_homo = gen_poisson_gp1_results_homo.predict(X_test_homo)
#gen_poisson_gp1_predictions is a pandas Series object that contains the predicted  count for 
#each row in the X_test matrix. Remember that y_test contains the actual observed counts.

#plot the predicted vs actual
gp_predicted_counts_homo = gen_poisson_gp1_predictions_homo
actual_counts_homo = y_test_homo['HomoRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual hate crimes GP model')
predicted, = plt.plot(X_test_homo.index, gp_predicted_counts_homo, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_homo.index, actual_counts_homo, 'ro-', label='Actual counts', alpha=0.5)
plt.legend(handles=[predicted, actual])
plt.show()

#make data frame for easy plotting

gp1_df_homo = pd.DataFrame({'Actual':actual_counts_homo , 'Predicted':gp_predicted_counts_homo })
gp1_df_homo.sort_values(['Predicted'], ascending=False)


ax = plt.gca()
gp1_df_homo.plot(kind='line',y='Actual',ax=ax)
gp1_df_homo.plot(kind='line',y='Predicted', color='red', alpha=0.5, ax=ax)
plt.title('Homophobic Hate Crime predictions VS actual')

gp1_df_homo.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


fig, ax = plt.subplots()
ax.scatter(actual_counts_homo, gp_predicted_counts_homo)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

ax = sns.scatterplot(x='Actual', y='Predicted', data= gp1_df_homo)


#%%   ALL TOGETHER SUPERSEDED  
#poisson reg - all PLACE TOGETHER VS total RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['TotalRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['TotalRate'].mean()))

#create train and test data frames. 
mask_all = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_all = finalhatedata_bng_nonNAN[mask_all]
test_all = finalhatedata_bng_nonNAN[~mask_all]

len(test_all)
len(train_all)
print('Training data set length='+str(len(train_all)))
print('Testing data set length='+str(len(test_all)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_all = """TotalRate ~ allfeatures_count"""

#Set up the X and y matrices for the training and testing data sets.
y_train_all, X_train_all = dmatrices(expr_all, train_all, return_type='dataframe')
y_test_all, X_test_all = dmatrices(expr_all, test_all, return_type='dataframe')

#train the model
poisson_training_results_all = sm.GLM(y_train_all, X_train_all, family=sm.families.Poisson()).fit()
print(poisson_training_results_all.summary())

poisson_predictions_all = poisson_training_results_all.get_prediction(X_test_all)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_all = poisson_predictions_all.summary_frame()
print(predictions_summary_frame_all)

#plot the predicted vs actual
predicted_counts_all=predictions_summary_frame_all['mean']
actual_counts_all = y_test_all['TotalRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual all hate crime incidents in London')
predicted, = plt.plot(X_test_all.index, predicted_counts_all, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_all.index, actual_counts_all, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for all hate crimes')
plt.scatter(x=predicted_counts_all, y=actual_counts_all, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')



#%% poisson reg - all PLACE separately  VS all RATE
import pandas as pd
import geopandas as gpd
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.formula.api import negativebinomial
import matplotlib.pyplot as plt
import seaborn as sns

#check the mean and variance 
print('variance='+str(finalhatedata_bng_nonNAN['TotalRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['TotalRate'].mean()))

#create train and test data frames. 
mask_all2 = np.random.rand(len(finalhatedata_bng_nonNAN)) < 0.8
train_all2 = finalhatedata_bng_nonNAN[mask_all2]
test_all2 = finalhatedata_bng_nonNAN[~mask_all2]

len(test_all2)
len(train_all2)
print('Training data set length='+str(len(train_all2)))
print('Testing data set length='+str(len(test_all2)))

list(finalhatedata_bng_nonNAN)

#set up the regression expression 
#expr = """AntiSemRat ~ synagogues_count  + temples_co + churches_c + synagogues"""
expr_all2 = """TotalRate ~ mosque_count + synagogues_count + disabledfeatures_count + faithfeatures_count + community_count + trans_count +lgbt_count  """

#Set up the X and y matrices for the training and testing data sets.
y_train_all2, X_train_all2 = dmatrices(expr_all2, train_all2, return_type='dataframe')
y_test_all2, X_test_all2 = dmatrices(expr_all2, test_all2, return_type='dataframe')

#train the model
poisson_training_results_all2 = sm.GLM(y_train_all2, X_train_all2, family=sm.families.Poisson()).fit()
print(poisson_training_results_all2.summary())

poisson_predictions_all2 = poisson_training_results_all2.get_prediction(X_test_all2)
#.summary_frame() returns a pandas DataFrame
predictions_summary_frame_all2 = poisson_predictions_all2.summary_frame()
print(predictions_summary_frame_all2)

#plot the predicted vs actual
predicted_counts_all2=predictions_summary_frame_all2['mean']
actual_counts_all2 = y_test_all2['TotalRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual all2 hate crime incidents in London')
predicted, = plt.plot(X_test_all2.index, predicted_counts_all2, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_all2.index, actual_counts_all2, 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for all2 hate crimes')
plt.scatter(x=predicted_counts_all2, y=actual_counts_all2, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')



#%% NEGATIVE BINOMIAL - ALL SEPARATELY VS TOTAL HATE CRIME

#try a NEGATIVE BINOMIAL DISTRIBUTION as the values are over dispersed 
#these prints access the LAMBDA created from the first poisson regression
print(poisson_training_results_all2.mu)
print(len(poisson_training_results_all2.mu))
import statsmodels.formula.api as smf

#Add the λ vector as a new column called ‘BB_LAMBDA’ to the Data Frame of the training data set.
train_all2['all2_LAMBDA'] = poisson_training_results_all2.mu

train_all2['AUX_OLS_DEP_all2'] = train_all2.apply(lambda x: ((x['TotalRate'] - x['all2_LAMBDA'])**2 - x['TotalRate']) / x['all2_LAMBDA'], axis=1)

NB_expr_all2 = """AUX_OLS_DEP_all2 ~ all2_LAMBDA - 1"""


NB_olsr_results_all2 = smf.ols(NB_expr_all2, train_all2).fit()
print(NB_olsr_results_all2.params)
#You will see the following single coefficient being printed out corresponding to the single regression 
#variable TotalAll_LAMBDA. This coefficient is the α that we were seeking
#we need to now work out if this value '1.262058' is significant 
#If the value of α is statistically not significant, then the Negative Binomial 
#regression model cannot do a better job of fitting the training data set than a Poisson regression model.

#The OLSResults object contains the t-score of the regression coefficient α 
NB_olsr_results_all2.tvalues

#From a t-value calculator, we can see that the critical t-value at a 99% confidence level (right-tailed),
# and degrees of freedom=160 is 2.34988. 
#therefore This is comfortably less than the t-statistic of α which we got which was  7.860347
#We conclude that a=1.262058 is statistically significant. 

#STEP 3: We supply the value of alpha found in STEP 2 into the statsmodels.genmod.families.family.NegativeBinomial class, 
#and train the NB2 model on the training data set. This is a one-step operation in statsmodels:

    
nb2_training_results_all2 = sm.GLM(y_train_all2, X_train_all2,family=sm.families.NegativeBinomial(alpha=NB_olsr_results_all2.params[0])).fit()
print(nb2_training_results_all2.summary())

#PREDICTIONS USING NB2

nb2_predictions_all2 = nb2_training_results_all2.get_prediction(X_test_all2)


nb2_predictions_summary_frame_all2 = nb2_predictions_all2.summary_frame()
print(nb2_predictions_summary_frame_all2)

#you can then plot these
nb2_predicted_counts_all2=nb2_predictions_summary_frame_all2['mean']
actual_counts_all2 = y_test_all2['TotalRate']
nb2_df_all2 = pd.DataFrame({'Actual':actual_counts_all2 , 'Predicted':nb2_predicted_counts_all2 })
nb2_df_all2

nb2_df_all2.sort_values(['Predicted'], ascending=False)

#nb2_df_all2 = nb2_df_all2.head(50)
sns.set(style="whitegrid")
sns.scatterplot(x='Actual', y='Predicted', data=nb2_df_all2)


#%% GENERALISED REG FOR LGBT  VS homophobic 
#generalised poisson as the mean and variance are not the same. 
#Build Consul’s Generalized Poison regression model, know as GP-1 for all hate crimes and faith features
    

#check the mean and variance 

print('variance='+str(finalhatedata_bng_nonNAN['TotalRate'].var()))
print('mean='+str(finalhatedata_bng_nonNAN['TotalRate'].mean()))

#the mean and variance are very different for total hate crime incidence. 

#use same mask, training and test data as for poisson . see above. 

#same regression expression as before for normal poisson. see above. 

gen_poisson_gp1_all2 = sm.GeneralizedPoisson(y_train_all2, X_train_all2, p=1)

#fit (train) the model
gen_poisson_gp1_results_all2 = gen_poisson_gp1_all2.fit()
print(gen_poisson_gp1_results_all2.summary())

#predict using the GP model. 
gen_poisson_gp1_predictions_all2 = gen_poisson_gp1_results_all2.predict(X_test_all2)
#gen_poisson_gp1_predictions is a pandas Series object that contains the predicted  count for 
#each row in the X_test matrix. Remember that y_test contains the actual observed counts.

#plot the predicted vs actual
gp_predicted_counts_all2 = gen_poisson_gp1_predictions_all2
actual_counts_all2 = y_test_all2['TotalRate']
fig = plt.figure()
fig.suptitle('Predicted versus actual hate crimes GP model')
predicted, = plt.plot(X_test_all2.index, gp_predicted_counts_all2, 'go-', label='Predicted counts')
actual, = plt.plot(X_test_all2.index, actual_counts_all2, 'ro-', label='Actual counts', alpha=0.5)
plt.legend(handles=[predicted, actual])
plt.show()

#make data frame for easy plotting

gp1_df_all2 = pd.DataFrame({'Actual':actual_counts_all2 , 'Predicted':gp_predicted_counts_all2 })
gp1_df_all2.sort_values(['Predicted'], ascending=False)


ax = plt.gca()
gp1_df_all2.plot(kind='line',y='Actual',ax=ax)
gp1_df_all2.plot(kind='line',y='Predicted', color='red', alpha=0.5, ax=ax)
plt.title('Total Hate Crime predictions VS actual')

gp1_df_all2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.show()


fig, ax = plt.subplots()
ax.scatter(actual_counts_all2, gp_predicted_counts_all2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

ax = sns.scatterplot(x='Actual', y='Predicted', data= gp1_df_all2)






#%%
#SAME BUT FOR THE BOROUGHS
#work out how many mosques per borough
#this provides an borough for every mosque
mosques_in_boroughs = gpd.sjoin(mosque_Gdf,boroughs_WGS84 , op='within')


#this then counts how many times each lsoa appears
count_of_mosques_in_boroughs = mosques_in_boroughs.groupby('NAME').size()
count_of_mosques_in_boroughs

#join the count to the original lsoa data 
boroughs_WGS84.columns
boroughs_WGS84.reset_index(inplace=True)
boroughs_WGS84_nameIndex = boroughs_WGS84.set_index('NAME')
boroughs_WGS84_nameIndex.index
boroughs_WGS84_nameIndex['mosque_count_per_borough'] = count_of_mosques_in_boroughs

boroughs_WGS84_nameIndex['mosque_count_per_borough'].sort_values(ascending=False)

# Set up figure and axis
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot the equal interval choropleth and add a legend
boroughs_WGS84_nameIndex.plot(column='mosque_count_per_borough', legend=True, axes=ax, colormap='Reds', linewidth=0.3)
boroughs_WGS84.plot(color='black', linewidth=0.3, ax=ax, alpha=0.5)
# Remove the axes
ax.set_axis_off()
# Set the title
ax.set_title("Equal Interval of Mosques in Boroughs's accross")
# Keep axes proportionate
plt.axis('equal')
# Draw map
plt.show()

#%% CLUSTERING. #PYPROJ PACKAGE NOT WORKING AT TIME OF WRITING CODE. 
import pysal as ps
#first create a spatial weights matrix

#use queen contiguity to for boundaries all around lsoa
spatialweights_queen = ps.queen_from_shapefile(lsoa, idVariable='LSOA11CD')


2+2
