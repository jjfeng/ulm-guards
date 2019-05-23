#!/usr/bin/env scons

# Simulate data under various different settings and fit models

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
import SCons.Script as sc

# Command line options

sc.AddOption('--clusters', type='string', help="Clusters to submit to. Default is local execution.", default='local')
sc.AddOption('--output', type='string', help="output folder", default='_output')

env = sc.Environment(
        ENV=os.environ,
        clusters=sc.GetOption('clusters'),
        output=sc.GetOption('output'))

sc.Export('env')

env.SConsignFile()

flag = 'simulation_do_no_harm'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_cost_decline'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_coverage_agg'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulation_misspecification'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'mimic_in_hospital_mortality'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'mimic_in_hospital_mortality_holdout'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'mimic_length_of_stay'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'mimic_length_of_stay_holdout'
sc.SConscript(flag + '/sconscript', exports=['flag'])

flag = 'mnist_holdout'
sc.SConscript(flag + '/sconscript', exports=['flag'])
