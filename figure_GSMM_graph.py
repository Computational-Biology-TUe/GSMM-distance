#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:01:54 2019

@author: acabbia
"""

import graph_tool as gt
from cobra.io import read_sbml_model

model_path = "/home/acabbia/Documents/Muscle_Model/models/merged_100_2class/L_MODEL1707110535.xml"
model = read_sbml_model(model_path)
outfolder= '/home/acabbia/out/distance/graph/'

def gt_net(model):
    
    # Extract nodes
    mets = [m.id for m in model.metabolites]
    rxns = [r.id for r in model.reactions]
    nodes = mets+rxns
    
    # make dict: {index:node_id}
    dd = { nodes[n] : n for n in range(len(nodes))}
    
    #Extract edges
    substrates = []
    products = []
    
    for r in model.reactions:
        for s in r.reactants:
            substrates.append((s.id, r.id))
        for p in r.products:
            products.append((r.id , p.id))

    edges = substrates+products
    
    # Translate edges_id -> edges_index
    edges_index = [ (dd[e[0]] , dd[e[1]]) for e in edges]
    
    #Initialize Graph
    G = gt.Graph()
    
    #Populate Graph
    G.add_edge_list(edges_index)
    
       
    #Extract largest network component
    l = gt.topology.label_largest_component(G)
    GG = gt.GraphView(G, vfilt= l)
    
    return GG


def gt_draw(gt_graph, title):
    b , cmap = gt.topology.is_bipartite(gt_graph, partition=True)
    if b:
        gt.draw.graph_draw(gt_graph,
                           vertex_fill_color = cmap,
                           bg_color=[1,1,1,1],
                           output_size=((1080,1080)),
                           fmt= 'png',
                           output=outfolder+title)    
    else:
        print('Graph is not bipartite.')
    return

G = gt_net(model)
gt_draw(G, 'try')