#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:20:54 2017

@author: aditya
"""
from bokeh.io import *
from bokeh.plotting import figure
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row
from bokeh.models import HoverTool
from bokeh.palettes import Spectral6
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn import cluster

output_file("HW3.html")

#Reading in Data
df = pd.read_csv('Wholesale customers data.csv')
df_np = df.values

#Clustering using sci-kit learn

k = 3
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(df_np)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#Plotting Milk vs Grocery
p = figure(plot_width=400, plot_height=400)
    
for i in range(k):
    # select only data observations with cluster label == i
    ds = df_np[np.where(labels==i)]
    
    #Add a circle renderer with a size, color, and alpha
    p.circle(ds[:,3],ds[:,4], size=20, color = Spectral6[i], 
          alpha=0.8, legend = "Clustered Values {}".format(i))
    
    # plot the centroids
    p.diamond_cross(centroids[i,3],centroids[i,4], color = Spectral6[i+3], 
          alpha=0.8, legend = "Centroid {}".format(i), size = 50)

#Customizing attributes 
p.title.text = "Milk vs Grocery"
p.legend.location = "top_left"
p.grid.grid_line_alpha=0
p.xaxis.axis_label = 'Milk'
p.yaxis.axis_label = 'Grocery'
p.legend.click_policy = "hide"
#----------------------------------------------------------------------------------------
#Plotting Frozen vs Deli
p1 = figure(plot_width=400, plot_height=400)
    
for i in range(k):
    # select only data observations with cluster label == i
    ds = df_np[np.where(labels==i)]
    
    #Add a circle renderer with a size, color, and alpha
    p1.circle(ds[:,3],ds[:,6], size=20, color = Spectral6[i], 
          alpha=0.8, legend = "Clustered Values {}".format(i))
    
    # plot the centroids
    p1.diamond_cross(centroids[i,3],centroids[i,6], color = Spectral6[i+3], 
          alpha=0.8, legend = "Centroid {}".format(i), size = 50)

#Customizing attributes 
p1.title.text = "Milk vs Frozen"
p1.legend.location = "top_left"
p1.grid.grid_line_alpha=0
p1.xaxis.axis_label = 'Milk'
p1.yaxis.axis_label = 'Frozen'
p1.legend.click_policy = "hide"
#----------------------------------------------------------------------------------------
#Plotting Milk vs Region
p2 = figure(plot_width=400, plot_height=400)
    
for i in range(k):
    # select only data observations with cluster label == i
    ds = df_np[np.where(labels==i)]
    
    #Add a circle renderer with a size, color, and alpha
    p2.circle(ds[:,3],ds[:,1], size=20, color = Spectral6[i], 
          alpha=0.8, legend = "Clustered Values {}".format(i))
    
    # plot the centroids
    p2.diamond_cross(centroids[i,3],centroids[i,1], color = Spectral6[i+3], 
          alpha=0.8, legend = "Centroid {}".format(i), size = 50)

#Customizing attributes 
p2.title.text = "Milk vs. Region"
p2.legend.location = "bottom_right"
p2.grid.grid_line_alpha=0
p2.xaxis.axis_label = 'Milk'
p2.yaxis.axis_label = 'Region'
p2.legend.click_policy = "hide"
grid = gridplot([[p, p1], [p2, None]])

# Show Results
show(grid)

# CLustering using DBSCAN-----------------------------------------------------------------
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

Z = linkage(df_np) #Using Euclidean distance for maximum Cophenetic Correlation Coefficient
c, coph_dists = cophenet(Z, pdist(df_np))
print(c)

#Viewing clustering
print(Z[:20])

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
#-----------------------------------------------------------------------------------------

