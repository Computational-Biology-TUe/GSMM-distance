#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:52:09 2019

@author: acabbia
"""
pd.set_option("display.precision", 10)

val = [] 
ts = []

for x in range(100):
    for y in range(100):
        val.append(K.values[x][y])
        ts.append(K.index.values[x]+'-'+K.columns.values[y])

ts.replace

        
new_K = pd.DataFrame()

new_K["value"] = val
new_K["color"] = ts

new_K = new_K[new_K.value != 1]

new_K['color'].replace('S-S','r',inplace=True)
new_K['color'].replace('S-L','b',inplace=True)
new_K['color'].replace('L-S','b',inplace=True)
new_K['color'].replace('L-L','g',inplace=True)

nn = pd.DataFrame({'Liver-Skin': new_K.groupby('color').get_group('b').value,
                   'Liver-Liver': new_K.groupby('color').get_group('g').value,
                   'Skin-Skin': new_K.groupby('color').get_group('r').value}).plot.hist(bins=100,figsize=(15,10), stacked = True)
    
nn.set_xlabel(xlabel= 'Weisfeiler-Lehmann subtree Kernel similarity score' ,fontsize = 15)

plt.show()