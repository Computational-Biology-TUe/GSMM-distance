#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:29:16 2019

@author: acabbia
"""

import os
import cobra
import pandas as pd
import grakel as gk
import numpy as np 
from numpy import unique
from itertools import permutations
from scipy.spatial.distance import pdist , jaccard , squareform 
from sklearn.cluster import AgglomerativeClustering , SpectralClustering
from sklearn.metrics import accuracy_score , confusion_matrix

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

def make_binary_mat(library_folder,ref_model):
    #returns 3 binary matrices containing info about wheter reaction/metabolite/gene[i] 
    # from parent model has beeen added in each "contextualized" model

    reactions_matrix = pd.DataFrame(index=[r.id for r in ref_model.reactions])
    metabolite_matrix = pd.DataFrame(index=[m.id for m in ref_model.metabolites])
    gene_matrix = pd.DataFrame(index=[g.id for g in ref_model.genes])

    for filename in sorted(os.listdir(library_folder)):
        model = cobra.io.read_sbml_model(library_folder+filename)
        print("loading:", model.name)
        rxns = []
        mets = []
        genes = []
        
        label = str(filename).split('.')[0]
        
        for r in ref_model.reactions:
            if r in model.reactions:
                rxns.append(1)
            else:
                rxns.append(0)
                
        for m in ref_model.metabolites:
            if m in model.metabolites:
                mets.append(1)
            else:
                mets.append(0)
                
        for g in ref_model.genes:
            if g in model.genes:
                genes.append(1)
            else:
                genes.append(0)
         
        reactions_matrix[label] = pd.Series(rxns).values
        metabolite_matrix[label] = pd.Series(mets).values
        gene_matrix[label] = pd.Series(genes).values
    
    print("Done!")
    return reactions_matrix, metabolite_matrix, gene_matrix

def CalculateAccuracy(y, y_hat):
    
    accuracy = 0
    bestP = []
    perm = permutations(unique(y))
    
    for p in perm:
        
        tr = dict(zip(p, list(range(len(unique(y))))))
        y_tr = np.array([tr[v] for v in y])
        
        testAccuracy = accuracy_score(y_tr,y_hat)
        
        if testAccuracy > accuracy:
            accuracy = testAccuracy   
            bestP.append((p,testAccuracy))
    
    P_df = pd.DataFrame(bestP)
    bestLabel = list(P_df.max()[0])
    
    
    inv_tr = dict(zip(list(range(len(unique(y)))),bestLabel))
    inv_y_hat = np.array([inv_tr[v] for v in y_hat])
    
    cm = confusion_matrix(y , inv_y_hat , bestLabel)
            
    return accuracy , bestLabel , cm

def HCClust(DM, trueLabel):
    
    HC = AgglomerativeClustering(n_clusters=len(trueLabel.unique()), affinity='precomputed', linkage='average').fit(DM)
    y_pred = HC.labels_
    
    accHC , bestLabHC, cmHC = CalculateAccuracy(trueLabel, y_pred)

    return accHC , bestLabHC , cmHC

def SCClust(DM, trueLabel):
    
    SC = SpectralClustering(n_clusters=len(trueLabel.unique()), affinity='precomputed').fit(1-DM)
    y_pred = SC.labels_
    
    accSC , bestLabSC, cmSC = CalculateAccuracy(trueLabel, y_pred)

    return accSC , bestLabSC , cmSC
 
#%%    
model_library_folder = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
ref_model_file = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'
models_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',sep = '\t').sort_values(by='organism')

models_taxonomy.fillna(method='bfill', axis=0, inplace=True)

### Replaces and aggregates classes with less than 10 samples into a new "Other" class
for c in ['phylum','oxygenstat', 'gram', 'mtype']:
    for s in list(models_taxonomy[c].value_counts()[models_taxonomy[c].value_counts()<10].index):
        models_taxonomy[c].replace(s,'Other', inplace=True)


#%%
##### 
# MAKE GK DM
####

graphList = []
label = []

for model_name in sorted(os.listdir(model_library_folder)):
    print('Loading', model_name)
    model = cobra.io.read_sbml_model(model_library_folder+model_name)
    label.append(model.name)
    g = modelNet(model)
    graphList.append(g)
    
print('Done')

GL = pd.DataFrame(list(zip(label, graphList)), columns = ['organism','graph'])

#compute GK similarity matrix
kernel = gk.WeisfeilerLehman(base_kernel = gk.VertexHistogram, normalize= True)
GK = pd.DataFrame(kernel.fit_transform(GL['graph'].values))
GK.columns = GK.index = label

# take inverse as distance matrix
DM_GK = 1 - GK

####
# MAKE JACCARD DM
###

# make binary matrices (rxn, mets and gene matrices)
ref_model = cobra.io.read_sbml_model(ref_model_file)
reactions_matrix, metabolite_matrix, gene_matrix = make_binary_mat(model_library_folder, ref_model)

# compute pw distance matrix
DM_JD = pd.DataFrame(squareform(pdist(reactions_matrix.T, metric = jaccard)), 
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)


#%%
#####
# Hierarchical Clustering (average linkage)
#####
from datetime import datetime
# Network Similarity

print("Graph kernel similarity")
print("================================================")

gk_acc = []

for c in ['phylum','oxygenstat', 'gram', 'mtype', 'metabolism']:
    
    #safety check to catch unwanted "NaN"'s
    models_taxonomy[c][models_taxonomy[c].isna()]=='Other'

    start = datetime.now()

    print('Clustering by:', c)
    acc, bestlabel, cm = HCClust(DM_GK, models_taxonomy[c])
    print('Accuracy:', acc)
    
    gk_acc.append(acc)
    
    print(' ')
    end = datetime.now()
    scriptTime = end - start
    print("Took:",scriptTime.total_seconds(),'s')
    
    print("================================================")

    
# Hierarchical Clustering (average linkage)
# Jaccard Similarity

print("Jaccard similarity")
print("================================================")

jd_acc = []

for c in ['phylum','oxygenstat', 'gram', 'mtype', 'metabolism']:
    
    #safety check to catch unwanted "NaN"'s
    models_taxonomy[c][models_taxonomy[c].isna()]=='Other'

    start = datetime.now()

    print('Clustering by:', c)
    acc, bestlabel, cm = HCClust(DM_JD, models_taxonomy[c])
    print('Accuracy:', acc)
    
    jd_acc.append(acc)
    
    print(' ')
    end = datetime.now()
    scriptTime = end - start
    print("Took:",scriptTime.total_seconds(),'s')
    
    print("================================================")
    
HC_clustering_results  = pd.DataFrame(index=['phylum','oxygenstat', 'gram', 'mtype', 'metabolism'])
HC_clustering_results['Network Similarity'] = gk_acc
HC_clustering_results['Reactions Similarity'] = jd_acc

HC_clustering_results.plot.bar(title= 'Hierarchical clustering: accuracy')
    
#%%
#####
# Spectral Clustering 
#####
print("Graph kernel similarity")
print("================================================")

gk_acc = []

for c in ['phylum','oxygenstat', 'gram', 'mtype', 'metabolism']:
    
    #safety check to catch unwanted "NaN"'s
    models_taxonomy[c][models_taxonomy[c].isna()]=='Other'

    start = datetime.now()

    print('Clustering by:', c)
    acc, bestlabel, cm = SCClust(DM_GK, models_taxonomy[c])
    print('Accuracy:', acc)
    
    gk_acc.append(acc)
    
    print(' ')
    end = datetime.now()
    scriptTime = end - start
    print("Took:",scriptTime.total_seconds(),'s')
    
    print("================================================")

    
# Hierarchical Clustering (average linkage)
# Jaccard Similarity

print("Jaccard similarity")
print("================================================")

jd_acc = []

for c in ['phylum','oxygenstat', 'gram', 'mtype', 'metabolism']:
    
    #safety check to catch unwanted "NaN"'s
    models_taxonomy[c][models_taxonomy[c].isna()]=='Other'

    start = datetime.now()

    print('Clustering by:', c)
    acc, bestlabel, cm = SCClust(DM_JD, models_taxonomy[c])
    print('Accuracy:', acc)
    
    jd_acc.append(acc)
    
    print(' ')
    end = datetime.now()
    scriptTime = end - start
    print("Took:",scriptTime.total_seconds(),'s')
    
    print("================================================")
    
SC_clustering_results  = pd.DataFrame(index=['phylum','oxygenstat', 'gram', 'mtype', 'metabolism'])
SC_clustering_results['Network Similarity'] = gk_acc
SC_clustering_results['Reactions Similarity'] = jd_acc

SC_clustering_results.plot.bar(title= 'Spectral clustering: accuracy')
