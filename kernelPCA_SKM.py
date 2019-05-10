#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:34:33 2019

@author: acabbia
"""

import os
import cobra
import pandas as pd
import seaborn as sns
import grakel as gk 
from sklearn.decomposition import KernelPCA

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def modelNet(model):
    #Returns a grakel.Graph object from a cobra.model object
    
    edges_in = []
    edges_out = []
    edges = []
    
    for r in model.reactions:
        # enumerate 'substrates -> reactions' edges
        substrates = [s.id for s in r.reactants]
        edges_in.extend([(s,r.id) for s in substrates])
        # enumerate 'reactions -> products' edges
        products = [p.id for p in r.products]
        edges_out.extend([(p,r.id) for p in products])
        
    # Join lists
    edges.extend(edges_in)
    edges.extend(edges_out) 
    
    #labels
    label_m = {m.id:m.name for m in model.metabolites}
    label_r = {r.id:r.name for r in model.reactions}
    label_nodes = label_m
    label_nodes.update(label_r)
    label_edges= {p:p for p in edges}
    
    g = gk.Graph(edges, node_labels=label_nodes, edge_labels=label_edges)
    return g

library_folder = '/home/acabbia/Documents/Muscle_Model/models/library_GEOmerge/'

graphList = []
label = []

for model_name in os.listdir(library_folder):
    print('Loading', model_name)
    label.append(model_name.split('_')[2])
    model = cobra.io.read_sbml_model(library_folder+model_name)
    g = modelNet(model)
    graphList.append(g)
print('Done')
    
kernel = gk.WeisfeilerLehman(base_kernel = gk.VertexHistogram, normalize= True)
K = pd.DataFrame(kernel.fit_transform(graphList))

# 2-D scatterplot
kpca = KernelPCA(kernel="precomputed", n_components=2 , n_jobs=-1)
X_kpca = kpca.fit_transform(K)

sns.scatterplot(x = X_kpca[:,0] , y = X_kpca[:,1], hue = label)

# 3-D scatterplot
kpca = KernelPCA(kernel="precomputed", n_components=3, n_jobs=-1)
X_kpca = kpca.fit_transform(K)

fig = pyplot.figure(figsize=(8,8))
ax = Axes3D(fig)

# make color label 
td = {'old.xml':  'red' , 'young.xml': 'blue'}
hue = [td[l] for l in label]
      
ax.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2] , c = hue )

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

ax.legend()
pyplot.show()