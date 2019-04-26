#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:36:32 2019
@author: acabbia
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from cobra.io import read_sbml_model
from cobra.util import create_stoichiometric_matrix
from sklearn.metrics import pairwise_distances 
import os

def prob_vertex(S):
    #the probability of a steadystate random walk through hyperedges on hypergraph G(V,E) visiting vertex vi
    
    # Create stoichiometric matrix
    #S = create_stoichiometric_matrix(model, array_type='DataFrame')
    
    #Create incidence matrix of the hypergraph
    H = np.zeros(S.shape)
    H[S!=0] = 1
    
    #Vertex degree
    d = H.sum(axis=1)
    
    #Probability of a steadystate random walk visiting vertex v
    Pg = d/d.sum()
    
    return Pg


def JSHK(S1, S2):
    
    Pg_1 = prob_vertex(S1)
    Pg_2 = prob_vertex(S2)
    
    K = np.log(2) - np.exp2(jensenshannon(Pg_1, Pg_2))
    
    return K

#%%    
##### Add S arrays to a list
    
library_folder = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'

S_mat_list = []
model_label = []

for filename in sorted(os.listdir(library_folder)):
    model = read_sbml_model(library_folder+filename)
    print("Loading:", model.name)
    model_label.append(model.name)
    S = create_stoichiometric_matrix(model)
    S_mat_list.append(S)
    
#%%
#Make array from list
S_mat_array = np.array(S_mat_list)        

##### compute JSHK Kernel matrix    
K_mat = pairwise_distances(S_mat_array, metric = JSHK)

#%%
