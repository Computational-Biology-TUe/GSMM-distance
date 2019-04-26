#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:21:50 2019

@author: acabbia
"""
import numpy as np
from cobra.io import read_sbml_model
from cobra.util import create_stoichiometric_matrix
from halp.directed_hypergraph import DirectedHypergraph
from scipy.stats import entropy

model1 = read_sbml_model('/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/Acidaminococcus_sp_D21.xml')
model2 = read_sbml_model('/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/Aeromonas_caviae_Ae398.xml')

def makeHypergraph(model):
    
    S = create_stoichiometric_matrix(model, array_type='DataFrame')
    H = DirectedHypergraph()
    
    for i in range(len(S.columns)):
        nodes_df = S.iloc[:,i][S.iloc[:,i]!=0]
        
        edge_name = nodes_df.name
        head_nodes = set(nodes_df[nodes_df>0].index)
        tail_nodes = set(nodes_df[nodes_df<0].index)
        
        H.add_hyperedge(head_nodes , tail_nodes, {'reaction' : edge_name})
        
        if model.reactions.get_by_id(edge_name).reversibility:
            
            H.add_hyperedge( tail_nodes, head_nodes, {'reaction' : edge_name+'_rev'})
            
    return H


def prob_vertex(model):
    #the probability of a steadystate random walk through hyperedges on hypergraph G(V,E) visiting vertex vi
    
    # Create stoichiometric matrix
    S = create_stoichiometric_matrix(model, array_type='DataFrame')
    
    #Create hypergraph incidence matrix
    H = np.zeros(S.shape)
    H[S!=0] = 1
    
    #Vertex degree
    d = H.sum(axis=1)
    
    #Probability of a steadystate random walk visiting vertex v
    Pg = d/d.sum()
    
    return Pg
'''
def compositeProba(model1, model2):
    
    Pg_1 = prob_vertex(model1)
    Pg_2 = prob_vertex(model2)
    
    alpha_1 = len(Pg_1)/(len(Pg_1)+len(Pg_2))
    alpha_2 = len(Pg_2)/(len(Pg_1)+len(Pg_2))
    
    Pu = alpha_1*Pg_1 +alpha_2*Pg_2
    
    return Pu
'''

def JSHK(model1, model2):
    
    Pg_1 = prob_vertex(model1)
    Pg_2 = prob_vertex(model2)
    
    alpha_1 = len(Pg_1)/(len(Pg_1)+len(Pg_2))
    alpha_2 = len(Pg_2)/(len(Pg_1)+len(Pg_2))
    
    
    K = np.log(2) - ((alpha_1 - 0.5) * entropy(Pg_1)) - ((alpha_2 - 0.5) * entropy(Pg_2))
    
    return K
    
