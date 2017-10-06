# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from bokeh.io import *
from bokeh.plotting import figure
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
from bokeh.models import HoverTool
from bokeh.palettes import Spectral6
import numpy as np

output_file("HW4.html")

#Reading in Data
df = pd.read_csv('Nutrition_data.csv')

# Exploratory Analysis of Data

#We observe that 54 observations against 1093 variables are available to us.
df.shape

# Observing various statistics from the dataset
df.describe() 

# Observations made (sample for our given population, owing to large set of features):
'''
- People generally eat sandwiches 3 times a week for breakfast (2.907) and the maximum number 
of sandwich intake reported is 9!
- Most of the breakfast diet is contained of sandwich, eggs, yoghurt and creamcheese- 
    people eat the above around 3 times a week.
- People have a higher intake of animal protein(mean = 48.7 units), as compared to 
vegetable protein (mean = 31.46).
'''

# Feature Selection using VarianceThreshold- only on numerical features-
#       Will remove features with a zero variance, i.e.- features which do not vary much.

# Obtaining indexes of all the numerical features in our dataset.
numericalIndex = []
for c in range(2,df.shape[0]):
    if str(type(df.iloc[:,c][0])) == "<class 'numpy.int64'>":
        print(type(df.iloc[:,c][0]))
        numericalIndex.append(c)
numdf = df.iloc[:, numericalIndex]

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold = 7)
sel.fit_transform(numdf)
numdf.shape

'''
We can observe here that even though we have set the variance threshold really high- 7,
there are no variables being eliminated due to lack of variance. This is mostly attributed
to the minimal sample space of observations available to us, which results in high variance
in our features. Thus, we are unable to eliminate features due to lack of variance. 
'''

# Converting all categorical variable to numerical variables to perform the same 
#   VarianceThreshold feature Engineering

df_columns = df.select_dtypes(['object']).columns

tempdf = df[:]
for index in df_columns:
    tempdf[index] = tempdf[index].astype('category')

tempdf[df_columns] = tempdf[df_columns].apply(lambda x: x.cat.codes)

# Applying Variance Threshold on entire dataset of 1000 variables now

#Confirming shape of our tempdf
tempdf.shape

i = 0.1
while tempdf.shape == df.shape:
    sel = VarianceThreshold(threshold = 10)
    sel.fit_transform(tempdf)
    i += 0.1
    
print("Our dataset changed at has most variables below {} variance.".format(i))
'''
 THE ABOVE LOOP DID NOT STOP FOR AWHILE! Our dataset has great variance amongst the 
features, thus we'll have to use some other form of feature engineering or proceed with 
our prediction modeling. 
'''

# USING PRINCIPAL COMPONENTS ANALYSIS and Clustering of our data

from sklearn.decomposition import PCA
    
pca = PCA(n_components=4)
pca.fit(tempdf)
PCA(copy=True, n_components=2, whiten=False)

df_4d = pca.transform(tempdf)

df_4d = pd.DataFrame(df_4d)
df_4d.index = tempdf.index

# Displaying newly computed components of our data
df_4d.columns = ['PC1','PC2','PC3','PC4']
df_4d.head()

# Displaying the variance ratio of our principal components
print(pca.explained_variance_ratio_) 

'''
Here we can observe that our first 2 components account for 89% abd 6.7% of the variance 
respectively, i.e.- around 96% of the entire variance in our data, thus we can use only 
first 2 components for basing our clustering mechanism on. 
'''

# Plotting Scatterplot of the 2 Main Principal Components with annotations.

#--------------------------PLOT-1--------------------------------------------------------
source = ColumnDataSource(dict(
    label=['Heart Disease' if bool == 0 else 'No Heart Disease' 
                              for bool in tempdf['heart_disease']],
    cancer=list(df['cancer']),
    diabetes=list(df['diabetes']),
    ever_smoked=list(df['ever_smoked'])
))
p = figure(plot_width=550, plot_height=550)

#Add a circle renderer with a size, color, and alpha
p.circle(list(df_4d['PC2']), list(df_4d["PC1"]), size=(tempdf["diabetes"]+1)*20, 
         color=[Spectral6[bool] for bool in tempdf['heart_disease']], 
         alpha=0.8, legend ='label', source = source)

#Customizing attributes 
p.title.text = "Belly Size & Heart Disease Occurence"

#p.legend.location = "bottom_right"
p.grid.grid_line_alpha=0
p.xaxis.axis_label = 'PC2'
p.yaxis.axis_label = 'PC1'
p.legend.click_policy = "hide"

# HoverTool Introduction
hover = HoverTool(tooltips=[
    ("Cancer", "@cancer"),
    ("Diabetes", "@diabetes"),
    ("Ever-Smoked", "@ever_smoked")
])

# Adding HoverTool to existing set of tools
p.add_tools(hover)

'''
Some of the observations made from Plot-1:
    - Most of our population has heart disease.
    - Most of the people with heart disease, DO NOT suffer from diabetes!(not very reliable)
'''
#--------------------------PLOT-2--------------------------------------------------------
source = ColumnDataSource(dict(
    label=['Left Handed' if bool == 1 else 'Right Handed' 
                              for bool in tempdf['left_hand']],
    Athiest=list(df['atheist']),
    donuts=list(df['GROUP_DONUTS_TOTAL_GRAMS']),
    ever_smoked=list(df['ever_smoked'])
))
p1 = figure(plot_width=550, plot_height=550)

#Add a circle renderer with a size, color, and alpha
p1.circle(list(df_4d['PC2']), list(df_4d["PC1"]), size=(tempdf["atheist"]+1)*20, 
         color=[Spectral6[bool+3] for bool in tempdf['left_hand']], 
         alpha=0.8, legend ='label', source = source)

#Customizing attributes 
p1.title.text = "Left & Right Handedness vs Athiest + Donut Consumption"

#p.legend.location = "bottom_right"
p1.grid.grid_line_alpha=0
p1.xaxis.axis_label = 'PC2'
p1.yaxis.axis_label = 'PC1'
p1.legend.click_policy = "hide"

# HoverTool Introduction
hover = HoverTool(tooltips=[
    ("Athiest", "@Athiest"),
    ("Donuts", "@donuts gm"),
    ("Hand", "@label")
])

# Adding HoverTool to existing set of tools
p1.add_tools(hover)

show(row(p,p1))

'''
Some of the observations made from Plot-1:
    - Most of our population has is Right-Handed.
    - Most of our population has an atheist majority, wherein most of those are also 
        right-handed. A good portion of left-handed people are also atheists.
'''

#----------------------------------------------------------------------------------------
# Correlation Matrix of our Columns:

corr = tempdf.corr()

from math import pi
from bokeh.models import (
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)

# reshape to 1D array or rates with a month and year for each row.
tempdf2 = pd.DataFrame(corr.stack(), columns=['variance']).reset_index()

colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=tempdf2.variance.min(), 
                           high=tempdf2.variance.max())

source = ColumnDataSource(tempdf2)

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(title="Correlation Matrix- 1093 variables",
           x_range=list(tempdf.columns), y_range=list(reversed(tempdf.columns)),
           x_axis_location="above", plot_width=900, plot_height=900,
           tools=TOOLS, toolbar_location='below')

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="level_0", y="level_1", width=1, height=1,
       source=source,
       fill_color={'field': 'variance', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')

p.select_one(HoverTool).tooltips = [
     ('Feature 1', '@level_0'),
     ('Feature 2', '@level_1'),
     ('Variance', '@variance'),
]

show(p)

'''
Observations made from this correlation matrix by box-zooming into the darker regions:
    - Vitamin A intake positively correlates with Omega 3 intake.
    - Fish Type directly correlates with Omega 3 (which makes sense as well as Fish is
    a major supply of Omega 3.)
    - More consumption of plain yoghurt also sees an increase in consumption of honey oats.
    - Consumption of Honey Oats bunces is slightly correlated with Honey flavored Cheerios.
    - Cocoa Krispy Pebble puffs strongly correlated with breakfast sandwiches with eggs or meat.
    - More observations could have been made if more metadata corresponding to description
    of features was provided, as most of the naming convention is not understood as such.
'''
