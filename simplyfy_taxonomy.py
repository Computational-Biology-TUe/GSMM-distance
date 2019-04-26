#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:29:23 2019

@author: acabbia
"""

### AGORA taxonomy : reduce number of classes 

import pandas as pd

models_taxonomy = pd.read_csv('/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv',sep = '\t').sort_values(by='organism')
models_taxonomy.fillna(method='bfill', axis=0, inplace=True)

for c in models_taxonomy.columns:
    print(models_taxonomy[c].value_counts())
    print("========================================")
    
### Replaces and aggregates classes with less than 10 samples into a new "Other" class
for c in ['phylum','oxygenstat', 'gram', 'mtype']:
    for s in list(models_taxonomy[c].value_counts()[models_taxonomy[c].value_counts()<10].index):
        models_taxonomy[c].replace(s,'Other', inplace=True)

for c in ['metabolism']:
    for s in list(models_taxonomy[c].value_counts()[models_taxonomy[c].value_counts()<14].index):
        models_taxonomy[c].replace(s,'Other', inplace=True)
        
models_taxonomy['metabolism'].replace('Saccharolytic, fermentative or respiratory','Saccharolytic, respiratory or fermentative', inplace=True)
models_taxonomy['metabolism'].replace('Respiration or fermentation of carbohydrates and central metabolism intermediates','Saccharolytic, respiratory or fermentative', inplace=True)
