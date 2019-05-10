#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:13:28 2019

@author: acabbia
"""

#AGORA universal model

import cobra
import os

model_library_folder = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'

universe = cobra.Model('AGORA_universe_model')

for model_name in os.listdir(model_library_folder):
    print('Loading', model_name)
    model = cobra.io.read_sbml_model(model_library_folder+model_name)
    for rxn in model.reactions:
        if rxn not in universe.reactions:
            universe.add_reaction(rxn)
            
cobra.io.write_sbml_model(universe, model_library_folder+'AGORA_universe.xml')