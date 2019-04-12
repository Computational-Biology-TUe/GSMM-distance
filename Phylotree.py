import os
import cobra
import pandas as pd
import grakel as gk 
from scipy.spatial.distance import pdist , jaccard , squareform 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree , TreeStyle

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



model_library_folder = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
ref_model_file = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'
models_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',sep = '\t').sort_values(by='organism')

#%%
#####
# MAKE REFERENCE NCBI TAXONOMY TREE
####
from ete3 import NCBITaxa

ncbi = NCBITaxa()
ncbi.update_taxonomy_database()

NCBI_ID = list(models_taxonomy['ncbiid'].dropna().values)
NCBI_tree = ncbi.get_topology(NCBI_ID)
NCBI_tree.set_outgroup(NCBI_tree.get_midpoint_outgroup())

NCBI_tree.write(format=1, outfile="/home/acabbia/Documents/Muscle_Model/GSMM-distance/NCBI_tree.nw")

#%%
##### 
# MAKE GK TREE
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

# Use 1-K as measure of Distance
DM_GK = DistanceMatrix(1-GK.values)

#make GK tree
sktree = nj(DM_GK, result_constructor=str)
GK_tree = Tree(sktree)

# style
ts = TreeStyle()
ts.show_leaf_name = True
ts.mode = "c"
ts.arc_start = -180 
ts.arc_span = 360

#plot tree
GK_tree.render(file_name='/home/acabbia/Documents/Muscle_Model/GSMM-distance/figures/GK_tree_AGORA.png', tree_style=ts)
GK_tree.show(tree_style=ts)

# save tree
GK_tree.write(format=1, outfile="/home/acabbia/Documents/Muscle_Model/GSMM-distance/GK_tree.nw")

#%%
####
# MAKE JACCARD TREE
###

# make binary matrices (rxn, mets and gene matrices)
ref_model = cobra.io.read_sbml_model(ref_model_file)
reactions_matrix, metabolite_matrix, gene_matrix = make_binary_mat(model_library_folder, ref_model)

# compute pw distance matrix
JD = pd.DataFrame(squareform(pdist(reactions_matrix.T, metric = jaccard)), 
                    index = reactions_matrix.columns, columns = reactions_matrix.columns)

DM_JD = DistanceMatrix(JD.values)

#make JD tree
sktree = nj(DM_JD, result_constructor=str)
JD_tree = Tree(sktree)
        
# style
ts = TreeStyle()
ts.show_leaf_name = True
ts.mode = "c"
ts.arc_start = -180 
ts.arc_span = 360

#plot tree
JD_tree.render(file_name='/home/acabbia/Documents/Muscle_Model/GSMM-distance/figures/JD_tree_AGORA.png', tree_style=ts)

# save tree
JD_tree.write(format=1, outfile="/home/acabbia/Documents/Muscle_Model/GSMM-distance/JD_tree.nw")

#%%
####
# Make FBA tree
####
sol_df = pd.DataFrame(index = [r.id for r in ref_model.reactions])

for filename in sorted(os.listdir(model_library_folder)):
    ####### load model     
    model = cobra.io.read_sbml_model(model_library_folder+filename)
    
    rxnlist = [r.id for r in model.reactions]
    (rxnlist[-1]) ## Biomass is always the last reaction in the list

    ###### set obj and (minimal) bounds
    model.objective= model.reactions.get_by_id(rxnlist[-1])
    
    #open all exchanges 
    for e in model.reactions:
        e.bounds = -1000,1000
    
    # Allow free exchange of oxygen and water, co2 and h+ outflow 
    model.reactions.get_by_id('EX_o2(e)').bounds= -1000,1000    # o2
    model.reactions.get_by_id('EX_h2o(e)').bounds= -1000,1000    # h2o
    model.reactions.get_by_id('EX_co2(e)').bounds=     0,1000    # co2
    model.reactions.get_by_id('EX_hco3(e)').bounds=     0,1000    # hco3
    model.reactions.get_by_id('EX_(e)').bounds=     0,1000    # h
    model.reactions.get_by_id('HMR_(e)').bounds=     0,100     # glc
    
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

# Use 1-K as measure of Distance
DM_FBA = DistanceMatrix(1-GK.values)

#make GK tree
sktree = nj(DM_FBA, result_constructor=str)
FBA_tree = Tree(sktree)

# style
ts = TreeStyle()
ts.show_leaf_name = True
ts.mode = "c"
ts.arc_start = -180 
ts.arc_span = 360

#plot tree
FBA_tree.render(file_name='/home/acabbia/Documents/Muscle_Model/GSMM-distance/figures/FBA_tree_AGORA.png', tree_style=ts)
FBA_tree.show(tree_style=ts)

# save tree
FBA_tree.write(format=1, outfile="/home/acabbia/Documents/Muscle_Model/GSMM-distance/FBA_tree.nw")

#%%
#Compare trees

FBA_tree.compare(GK_tree)


