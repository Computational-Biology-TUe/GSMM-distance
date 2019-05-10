#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:55:17 2019

@author: acabbia
"""
import os
import cobra 
import pandas as pd
import grakel as gk
import seaborn as sns
from matplotlib import pyplot as plt

def binary(model, ref_model):
       
    # init
    rxns = []
    mets = []
    genes = []

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
    
    return rxns, mets, genes

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

def FBA(model, ref_model):    
    
        
    ###### set obj and (minimal) bounds
    model.objective = model.reactions.get_by_id([r.id for r in model.reactions if 'biomass' in r.id][0])
    
    #open all exchanges 
    for e in model.reactions:
        e.bounds = -1000,1000
    
    # optimize (loopless)
    # sol = cobra.flux_analysis.loopless.loopless_solution(model)
    
    # optimize (normal FBA)
    sol = model.optimize()
    
    # flux distributions are appended following the index
    return sol.fluxes     
    

def load_library(path, ref_model_path):
    
    '''
    loads models from library folder and prepares data structures for further analysis
    returns:
        
        - Binary matrices (rxn,met,genes) --> EDA and Jaccard
        - Graphlist --> Graph Kernels
        - Flux vectors matrix --> cosine similarity 
    
    '''
 
    ref_model = cobra.io.read_sbml_model(ref_model_path)
    
    # Init
    reactions_matrix = pd.DataFrame(index = [r.id for r in ref_model.reactions])
    metabolite_matrix = pd.DataFrame(index = [m.id for m in ref_model.metabolites])
    gene_matrix = pd.DataFrame(index = [g.id for g in ref_model.genes])
    sol_df = pd.DataFrame(index = [r.id for r in ref_model.reactions])
    graphlist = []
        
    for filename in sorted(os.listdir(path)):
        model = cobra.io.read_sbml_model(path+filename)
        label = str(filename).split('.')[0]
                
        print("loading:", label)
        
        # 1: make binary matrices           
        rxns, mets, genes = binary(model, ref_model)
        reactions_matrix[label] = rxns
        metabolite_matrix[label] = mets
        gene_matrix[label] = genes

        # 2: make graphlist
        graphlist.append(modelNet(model))
    
        # 3: make flux matrix
        fluxes = FBA(model, ref_model)
        sol_df[label] = fluxes
        
    return reactions_matrix , metabolite_matrix , gene_matrix , graphlist , sol_df

#%%
    
path_PDGSMM = '/home/acabbia/Documents/Muscle_Model/models/merged_100/'
path_AGORA = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'

path_ref_PDGSMM = '/home/acabbia/Documents/Muscle_Model/models/HMR2.xml'
path_ref_AGORA = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'


# create labels
label_PDGSM = [s.split('_')[0] for s in sorted(os.listdir(path_PDGSMM))]


AGORA_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',
                 sep = '\t').sort_values(by='organism')

AGORA_taxonomy.fillna(method='bfill', axis=0, inplace=True)

### Replaces and aggregates classes with less than 10 samples into a new "Other" class
for c in ['phylum','oxygenstat', 'gram', 'mtype', 'metabolism']:
    for s in list(AGORA_taxonomy[c].value_counts()[AGORA_taxonomy[c].value_counts()<10].index):
        AGORA_taxonomy[c].replace(s,'Other', inplace=True)


label_AGORA_phylum = list(AGORA_taxonomy['phylum'].values)
label_AGORA_oxy = list(AGORA_taxonomy['oxygenstat'].values)
label_AGORA_gram = list(AGORA_taxonomy['gram'].values)
label_AGORA_type = list(AGORA_taxonomy['mtype'].values)
label_AGORA_nrg = list(AGORA_taxonomy['metabolism'].values)

#%%
rxns_PDGSM , met_PDGSM , gene_PDGSM , graphlist_PDGSM , flux_PDGSM = load_library(path_PDGSMM , path_ref_PDGSMM)    
rxns_AGORA , met_AGORA , gene_AGORA , graphlist_AGORA , flux_AGORA = load_library(path_AGORA  , path_ref_AGORA)    
    
#%%    

def boxplots(df, label):
    # Reactions/metabolites/genes content of the models, grouped by label

    groups = df.T.sum(axis=1).groupby(label)

    names = []
    data = []

    for g in groups:
        names.append(g[0])
        data.append(g[1].values)
        
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(labels = names,rotation=90)

#   ax.get_figure().savefig(outfolder+'boxplots/'+c+'.png', dpi=1200, bbox_inches='tight')
    plt.show()


# Explorative Data Analysis (boxplots)
# Reactions/metabolites/genes content of the models, grouped by label

boxplots(rxns_AGORA, label_AGORA_gram)
boxplots(rxns_AGORA, label_AGORA_oxy)
boxplots(rxns_AGORA, label_AGORA_phylum)
boxplots(rxns_AGORA, label_AGORA_type)
boxplots(rxns_PDGSM, label_PDGSM)

#%%
