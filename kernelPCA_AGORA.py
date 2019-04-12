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
import numpy as np
from sklearn.decomposition import KernelPCA

from matplotlib import pyplot as plt
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


model_library_folder = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
models_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',sep = '\t')

# Reformat organism name to match 
models_taxonomy['organism'] = [str(s).split(' ')[0]+' '+str(s.split(' ')[1].replace('.',' ')) for s in models_taxonomy['organism'].values]

graphList = []
label = []

for model_name in os.listdir(model_library_folder):
    print('Loading', model_name)
    label.append(model_name.split('_')[0]+' '+model_name.split('_')[1])
    model = cobra.io.read_sbml_model(model_library_folder+model_name)
    g = modelNet(model)
    graphList.append(g)
print('Done')


# Sort graphs by model name, to match with Taxonomy df
GL = pd.DataFrame(list(zip(label, graphList)), columns = ['organism','graph'])
GL = GL.sort_values(by = 'organism').reset_index()

kernel = gk.WeisfeilerLehman(base_kernel = gk.VertexHistogram, normalize= True)
K = pd.DataFrame(kernel.fit_transform(GL['graph'].values))

# K-PCA
# 2-D scatterplot
kpca = KernelPCA(kernel="precomputed", n_components=2 , n_jobs=-1)
X_kpca = kpca.fit_transform(K)

for c in ['phylum', 'mclass', 'order', 'family', 'genus','oxygenstat', 'gram']:
    g = sns.scatterplot(x = X_kpca[:,0] , y = X_kpca[:,1], hue = models_taxonomy[c].values, legend = 'brief')
    box = g.get_position() # get position of figure
    g.set_position([box.x0, box.y0, box.width * 1.25, box.height]) # resize position
    # Put a legend to the right side
    plt.legend(loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1)
    plt.show(g)


# 3-D scatterplot    
kpca = KernelPCA(kernel="precomputed", n_components=3, n_jobs=-1)
X_kpca = kpca.fit_transform(K)


for c in ['phylum', 'mclass', 'order','oxygenstat', 'gram']:
    
    cmap = plt.get_cmap('Set1')
    names = models_taxonomy[c].unique()
    colors = cmap(np.linspace(0, 1, len(names)))
    td = dict(zip(names,colors))

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')  
  
    for g in models_taxonomy.groupby(c):
        
        x = X_kpca[g[1].index][:,0]
        y = X_kpca[g[1].index][:,1]
        z = X_kpca[g[1].index][:,2]  
                
        ax.scatter(x, y, z, label = g[0] , c = td[g[0]])

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    plt.legend(loc='center right', bbox_to_anchor=(1.6, 0.5), ncol=1)
    plt.show()




































