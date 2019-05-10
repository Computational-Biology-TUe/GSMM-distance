#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:36:56 2018

@author: acabbia
"""
import cobra 
from sklearn.cluster import AgglomerativeClustering , SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np 
import seaborn as sns
import grakel as gk
from scipy.spatial.distance import pdist , jaccard , squareform , hamming

from matplotlib import pyplot as plt
from datetime import datetime
import os
import PIL

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
    
    return reactions_matrix, metabolite_matrix, gene_matrix

def D_to_K(D):
    # convert distance matrix into Kernel (similarity) matrix
    K = np.exp(-D)
    return K

def K_to_D(K):
    # convert kernel (similarity) matrix into distance matrix
    D = 1 - K
    return D

def savefig(plot, outfolder , title):
    plot.get_figure().savefig(outfolder+title, dpi=1200, bbox_inches='tight')

def png2tif(outfolder):
    for f in os.listdir(outfolder):
        tif = PIL.Image.open(outfolder+f)
        tif.save(outfolder+f[0:-3]+'tif')
           
def pw_dist_hist(M,metric):
    # plots hist of Kernel or Distance matrix
    
    # arguments:
    # M: (DataFrame) either K or D
    
    #filter D = 0 and K=1 (identities)
    M = M[M!=0]
    M = M[M!=1]
    # reshape and plot hist
    M = M.values.reshape(10000,-1)
      
    ax = pd.DataFrame(M).plot.hist(bins=100,figsize =(15,10),fontsize = 15, legend = False)
    
    ax.set_xlabel(xlabel=metric,fontsize = 15)
    
    show()    
    
    return ax

def clust(D,K,label):
    # performs clustering (SC and HC) prints accuracy of retrieval of original labels
    
    # arguments:
    # K = Kernel matrix
    # D = Distance matrix
    # label = (list) class label for each model
    
    # Agglomerative (hierarchical) clustering
    acc_df = pd.DataFrame()
    acc_HC = []
    acc_SC = []
    
    for r in range(0,10):
        #Agglomerative
        agg =  AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average').fit(D)
        y_pred = agg.labels_
        acc_HC.append(accuracy_score(label, y_pred))
      
        # Spectral Clustering 
        db = SpectralClustering(n_clusters=2, affinity = 'precomputed').fit(K)
        y_pred = db.labels_
        acc_SC.append(accuracy_score(label, y_pred))
        
    acc_df['HC'] = acc_HC
    acc_df['SC'] = acc_SC
        
    print("Accuracy HC:", str(acc_df['HC'].mean().round(2)), 'error:', acc_df['HC'].std().round(2))
    print("Accuracy SC:", str(acc_df['SC'].mean().round(2)), 'error:', acc_df['SC'].std().round(2))
    
        
def classify(D,K,label):
    # performs (10-fold CV) classification (with SVM and KNN), prints accuracy of retrieval of original labels
    
    # arguments:
    # K = Kernel matrix
    # D = Distance matrix
    # label = (list) class label for each model
    
    # K_NN 10-Fold CV
    neigh = KNeighborsClassifier(n_neighbors=3, metric = 'precomputed')
    scores_K_NN = cross_val_score(neigh, D.values,label, cv = 10, scoring = 'accuracy')
    print("Accuracy K-NN:", str(scores_K_NN.mean().round(2)), 'CV error:', scores_K_NN.std().round(2)) 
    
    # Kernel SVM 10-fold CV
    clf = SVC(kernel='precomputed', C=1)
    scores_K_SVM = cross_val_score(clf, K.values,label, cv = 10, scoring = 'accuracy')
    print("Accuracy SVM:", str(scores_K_SVM.mean().round(2)),'CV error:', scores_K_NN.std().round(2))

# paths
outfolder = '/home/acabbia/Documents/Muscle_Model/GSMM-distance/figures/'
library_folder= '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
ref_model_file = library_folder + "/AGORA_universe.xml"

models_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',sep = '\t')

#%%
##################################################################################################################### 

## Part 1: Dataset EDA 
print('Part 1: Exploratory analysis')    

# make binary matrices (rxn,mets and gene matrices)
ref_model = cobra.io.read_sbml_model(ref_model_file)
reactions_matrix, metabolite_matrix, gene_matrix = make_binary_mat(library_folder, ref_model)

# make boxplots of model content, grouped by label

def boxplots(df):
    for c in  ['phylum', 'mclass', 'order','oxygenstat', 'gram']:
        rr = df.T.sum(axis=1).groupby(models_taxonomy[c].values)
    
        names = []
        data = []
        for g in rr:
            names.append(g[0])
            data.append(g[1].values)
            
        ax = sns.boxplot(data=data)
        ax.set_xticklabels(labels = names,rotation=90)
        ax.get_figure().savefig(outfolder+'boxplots/'+c+'.png', dpi=1200, bbox_inches='tight')
        plt.show()


boxplots(metabolite_matrix)
boxplots(reactions_matrix)
#####################################################################################################################

## Part 2: Metabolic reconstructions
print('Part 2: Metabolic Reconstructions')

metric="Jaccard metric"
print(metric)

# Jaccard metric 
start = datetime.now()

pw_R = pd.DataFrame(squareform(pdist(reactions_matrix.T, metric = jaccard)), 
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)

pw_m = pd.DataFrame(squareform(pdist(metabolite_matrix.T, metric = jaccard)), 
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)

D = (pw_R + pw_m)/2
label = [dd[l.split('_')[0]] for l in D.columns]

end = datetime.now()
scriptTime = end - start
print("Runtime:",scriptTime.total_seconds(),'s')

#plot metric distribution and save fig
plot = pw_dist_hist(D,metric)
savefig(plot,outfolder,'jaccard.png')

# clustering and classification
K = D_to_K(D)
clust(D,K,label)
classify(D,K,label)

print("========================================================================================================")
metric="Hamming metric"
print(metric)

# Hamming metric 
start = datetime.now()

pw_R = pd.DataFrame(squareform(pdist(reactions_matrix.T, metric = hamming)), 
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)

pw_m = pd.DataFrame(squareform(pdist(metabolite_matrix.T, metric = hamming)),
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)

D = (pw_R + pw_m)/2
label = [dd[l.split('_')[0]] for l in D.columns]

end = datetime.now()
scriptTime = end - start
print("Runtime:",scriptTime.total_seconds(),'s')

#plot metric distribution and save fig
plot = pw_dist_hist(D,metric)
savefig(plot,outfolder,'hamming.png')

# clustering and classification
K = D_to_K(D)
clust(D,K,label)
classify(D,K,label)
print("========================================================================================================")
#####################################################################################################################

# Part 2: Graphs
print('Part 3: Graph Topology')    

# Build list of grakel.Graph object from cobra models

graphList = []
label = []

for model_name in os.listdir(library_folder):
    label.append(dd[model_name.split('_')[0]])
    model = cobra.io.read_sbml_model(library_folder+model_name)
    g = modelNet(model)
    graphList.append(g)

## Classification (10-fold CrossVal)

# Kernel functions to be used     
fn = [gk.WeisfeilerLehman, gk.NeighborhoodSubgraphPairwiseDistance]

for f in fn:
    
    try:
        gkernel = f(base_kernel = gk.VertexHistogram, normalize= True)
        name = 'K_WLS.png'
    except:
        gkernel = f(normalize = True)
        name = 'K_NSPD.png'
        
    metric = str(gkernel).split('(')[0] + " Kernel"
    print(metric)
    
    # Calculate the kernel (Gram) matrix.
    start = datetime.now()
    K = pd.DataFrame(gkernel.fit_transform(graphList))
    end = datetime.now()
    scriptTime = end - start
    print("Runtime:",scriptTime.total_seconds(),'s')
    
    #plot metric distribution and save fig
    plot = pw_dist_hist(K,metric)
    savefig(plot,outfolder,name)
    
    # clustering and classification
    D = K_to_D(K)
    clust(D,K,label)
    classify(D,K,label)
    print("========================================================================================================")

#####################################################################################################################
### Part 3 (models)
print('Part 4: Constraint-based models')    
metric="Cosine similarity"
print(metric) 
start = datetime.now()
#initialize flux distribution DF
sol_df = pd.DataFrame(index = [r.id for r in ref_model.reactions])

for filename in os.listdir(library_folder):
    ####### load model     
    model = cobra.io.read_sbml_model(library_folder+filename)
         
    ###### set obj and (minimal) bounds
    model.objective= 'HCC_biomass'
    '''
    model.objective = 'HMR_6916'  # ATPS4m
    '''
    #open all exchanges 
    for e in model.reactions:
        e.bounds = -1000,1000
    '''
    # Allow free exchange of oxygen and water, co2 and h+ outflow 
    model.reactions.get_by_id('HMR_9048').bounds= -1000,1000    # o2
    model.reactions.get_by_id('HMR_9047').bounds= -1000,1000    # h2o
    model.reactions.get_by_id('HMR_9058').bounds=     0,1000    # co2
    model.reactions.get_by_id('HMR_9078').bounds=     0,1000    # hco3
    model.reactions.get_by_id('HMR_9079').bounds=     0,1000    # h
    model.reactions.get_by_id('HMR_9034').bounds=     0,100     # glc
    '''
    # optimize (loopless)
    sol = cobra.flux_analysis.loopless.loopless_solution(model)
    
    # optimize (normal FBA)
    sol = model.optimize()
    
    # flux distributions are appended following the index
    sol_df[filename] = sol.fluxes 

# Fluxes below tolerance (1e-9) are considered to be zero 
sol_df = sol_df.replace(np.nan,0)
sol_df[sol_df<1e-9] = 0
processed = sol_df.T

K = pd.DataFrame(cosine_similarity(processed))
K.index = K.columns = processed.index
label = [dd[l.split('_')[0]] for l in K.columns]

end = datetime.now()
scriptTime = end - start
print('Took:', scriptTime.total_seconds(),'s')

#plot metric distribution and save fig

plot = pw_dist_hist(K,metric)
savefig(plot,outfolder,'cosine.png')

# clustering and classification
D = K_to_D(K)
clust(D,K,label)
classify(D,K,label)
print("========================================================================================================")