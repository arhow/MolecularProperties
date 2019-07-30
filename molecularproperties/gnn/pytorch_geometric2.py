# from common import *
from collections import defaultdict
import os
#numerical libs
import math
import numpy as np
import random
import PIL
import cv2
import matplotlib
# matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg
# print('matplotlib.get_backend : ', matplotlib.get_backend())
#print(matplotlib.__version__)

from dscribe.descriptors import ACSF
from dscribe.core.system import System

# torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel

from torch.nn.utils.rnn import *


# std libs
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp

#from pprintpp import pprint, pformat
import json
import zipfile



# import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import rdkit.Chem.Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
DrawingOptions.bondLineWidth=1.8

from rdkit.Chem.rdmolops import SanitizeFlags
if 0: # Supress warnings from rdkit
    from rdkit import rdBase
    from rdkit import RDLogger
    rdBase.DisableLog('rdApp.error')
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)


import scipy
from sklearn import preprocessing
## feature extraction #####################################################


COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_COUPLING_TYPE)]


#---

SYMBOL=['H', 'C', 'N', 'O', 'F']

BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATION=[
    #Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    #Chem.rdchem.HybridizationType.SP3D,
    #Chem.rdchem.HybridizationType.SP3D2,
]

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    if 0:
        if sum(one_hot)==0: print('one_hot_encoding() return NULL!', x, set)
    return one_hot


'''
node_feature :
    category 
        one_hot_encoding(symbol,SYMBOL)  #5 
        (acceptor,) #1
        (donor,   ) #1
        (aromatic,) #1
        one_hot_encoding(hybridization,HYBRIDIZATION) #3
    real  
        (num_h,  ) #1
        (atomic, ) #1
        
        
edge_feature :
    category 
        one_hot_encoding(bond_type,BOND_TYPE)  #4 
    real  
        np.digitize(distance,DISTANCE) #1
        
'''



class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__.keys())
    
###############################################################
def compute_kaggle_metric( predict, coupling_value, coupling_type):

    mae     = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)

            mae[t] = m
            log_mae[t] = log_m
        else:
            pass

    return mae, log_mae



###############################################################
def make_graph(molecule_name, gb_structure, gb_scalar_coupling, ):
    #https://stackoverflow.com/questions/14734533/how-to-access-pandas-groupby-dataframe-by-key

    #----
    df = gb_scalar_coupling.get_group(molecule_name)
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type',
    #        'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'],

    coupling = Struct(
        id = df.id.values,
        contribution = df[['fc', 'sd', 'pso', 'dso']].values,
        index = df[['atom_index_0', 'atom_index_1']].values,
        #type = np.array([ one_hot_encoding(t,COUPLING_TYPE) for t in df.type.values ], np.uint8)
        type = np.array([ COUPLING_TYPE.index(t) for t in df.type.values ], np.int32),
        value = df.scalar_coupling_constant.values,
    )


    #----
    df = gb_structure.get_group(molecule_name)
    df = df.sort_values(['atom_index'], ascending=True)
    # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
    a   = df.atom.values.tolist()
    xyz = df[['x','y','z']].values
    mol = mol_from_axyz(a, xyz)


    #---
    assert( #check
       a == [ mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    )

    #---
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    ## ** node **
    #[ a.GetSymbol() for a in mol.GetAtoms() ]

    num_atom = mol.GetNumAtoms()
    symbol   = np.zeros((num_atom,len(SYMBOL)),np.uint8) #category
    acceptor = np.zeros((num_atom,1),np.uint8)
    donor    = np.zeros((num_atom,1),np.uint8)
    aromatic = np.zeros((num_atom,1),np.uint8)
    hybridization = np.zeros((num_atom,len(HYBRIDIZATION)),np.uint8)
    num_h  = np.zeros((num_atom,1),np.float32)#real
    atomic = np.zeros((num_atom,1),np.float32)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i]        = one_hot_encoding(atom.GetSymbol(),SYMBOL)
        aromatic[i]      = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(atom.GetHybridization(),HYBRIDIZATION)

        num_h[i]  = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()

    #[f.GetFamily() for f in feature]
    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    ## ** edge **
    num_edge = num_atom*num_atom - num_atom
    edge_index = np.zeros((num_edge,2), np.uint8)
    bond_type  = np.zeros((num_edge,len(BOND_TYPE)), np.uint8)#category
    distance   = np.zeros((num_edge,1),np.float32) #real
    angle      = np.zeros((num_edge,1),np.float32) #real

    norm_xyz = preprocessing.normalize(xyz, norm='l2')

    ij=0
    for i in range(num_atom):
        for j in range(num_atom):
            if i==j: continue
            edge_index[ij] = [i,j]

            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type[ij] = one_hot_encoding(bond.GetBondType(),BOND_TYPE)

            distance[ij] = ((xyz[i] - xyz[j])**2).sum()**0.5
            angle[ij] = (norm_xyz[i]*norm_xyz[j]).sum()

            ij+=1
    ##-------------------

    graph = Struct(
        molecule_name = molecule_name,
        smiles = Chem.MolToSmiles(mol),
        axyz = [a,xyz],

        node = [symbol, acceptor, donor, aromatic, hybridization, num_h, atomic,],
        edge = [bond_type, distance, angle],
        edge_index = edge_index,

        coupling = coupling,
    )
    return graph






## xyz to mol #############################################################
#<todo> check for bug ????
# https://github.com/jensengroup/xyz2mol

def get_atom(atom):
    ATOM = [ x.strip() for x in ['h ','he', \
          'li','be','b ','c ','n ','o ','f ','ne', \
          'na','mg','al','si','p ','s ','cl','ar', \
          'k ','ca','sc','ti','v ','cr','mn','fe','co','ni','cu', \
          'zn','ga','ge','as','se','br','kr', \
          'rb','sr','y ','zr','nb','mo','tc','ru','rh','pd','ag', \
          'cd','in','sn','sb','te','i ','xe', \
          'cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy', \
          'ho','er','tm','yb','lu','hf','ta','w ','re','os','ir','pt', \
          'au','hg','tl','pb','bi','po','at','rn', \
          'fr','ra','ac','th','pa','u ','np','pu'] ]
    atom = atom.lower()
    return ATOM.index(atom) + 1

def getUA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence,valence) in enumerate(zip(maxValence_list, valence_list)):
        if maxValence - valence > 0:
            UA.append(i)
            DU.append(maxValence - valence)
    return UA,DU


def get_BO(AC,UA,DU,valences,UA_pairs,quick):
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i,j in UA_pairs:
            BO[i,j] += 1
            BO[j,i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = getUA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA,AC,quick)[0]

    return BO


def valences_not_too_large(BO,valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences,number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def BO_is_OK(BO,AC,charge,DU,atomic_valence_electrons,atomicNumList,charged_fragments):
    Q = 0 # total charge
    q_list = []
    if charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i,atom in enumerate(atomicNumList):
            q = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i,:]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    if (BO-AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):
        return True
    else:
        return False


def get_atomic_charge(atom,atomic_valence_electrons,BO_valence):
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge

def clean_charges(mol):
# this hack should not be needed any more but is kept just in case
#

    rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
                  '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
                  '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
                  '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
                  '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    fragments = Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)

    for i,fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol,fragment)

    return mol


def BO2mol(mol,BO_matrix, atomicNumList,atomic_valence_electrons,mol_charge,charged_fragments):
# based on code written by Paolo Toscani

    l = len(BO_matrix)
    l2 = len(atomicNumList)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
            '{1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)
    mol = rwMol.GetMol()

    if charged_fragments:
        mol = set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge)
    else:
        mol = set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences)

    return mol

def set_atomic_charges(mol,atomicNumList,atomic_valence_electrons,BO_valences,BO_matrix,mol_charge):
    q = 0
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i,:]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    q += 1
                    charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                    q += 2
                    charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    # shouldn't be needed anymore bit is kept just in case
    #mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol,atomicNumList,atomic_valence_electrons,BO_valences):
# The number of radical electrons = absolute atomic charge
    for i,atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom,atomic_valence_electrons[atom],BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol

def get_bonds(UA,AC):
    bonds = []

    for k,i in enumerate(UA):
        for j in UA[k+1:]:
            if AC[i,j] == 1:
                bonds.append(tuple(sorted([i,j])))

    return bonds

def get_UA_pairs(UA,AC,quick):
    bonds = get_bonds(UA,AC)
    if len(bonds) == 0:
        return [()]

    if quick:
        G=nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA)/2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]
 #           if quick and max_atoms_in_combo == 2*int(len(UA)/2):
 #               return UA_pairs
        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs

def AC2BO(AC,atomicNumList,charge,charged_fragments,quick):
    # TODO
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4,3]
    atomic_valence[8] = [2,1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5,4,3]
    atomic_valence[16] = [6,4,2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]


    atomic_valence_electrons = {}
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7

# make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    for atomicNum in atomicNumList:
        valences_list_of_lists.append(atomic_valence[atomicNum])

# convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = list(itertools.product(*valences_list_of_lists))

    best_BO = AC.copy()

# implemenation of algorithm shown in Figure 2
# UA: unsaturated atoms
# DU: degree of unsaturation (u matrix in Figure)
# best_BO: Bcurr in Figure
#

    for valences in valences_list:
        AC_valence = list(AC.sum(axis=1))
        UA,DU_from_AC = getUA(valences, AC_valence)

        if len(UA) == 0 and BO_is_OK(AC,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
            return AC,atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA,AC,quick)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC,UA,DU_from_AC,valences,UA_pairs,quick)
            if BO_is_OK(BO,AC,charge,DU_from_AC,atomic_valence_electrons,atomicNumList,charged_fragments):
                return BO,atomic_valence_electrons

            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO,valences):
                best_BO = BO.copy()

    return best_BO,atomic_valence_electrons


def AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick):
# convert AC matrix to bond order (BO) matrix
    BO,atomic_valence_electrons = AC2BO(AC,atomicNumList,charge,charged_fragments,quick)

# add BO connectivity and charge info to mol object
    mol = BO2mol(mol,BO, atomicNumList,atomic_valence_electrons,charge,charged_fragments)

    return mol


def get_proto_mol(atomicNumList):
    mol = Chem.MolFromSmarts("[#"+str(atomicNumList[0])+"]")
    rwMol = Chem.RWMol(mol)
    for i in range(1,len(atomicNumList)):
        a = Chem.Atom(atomicNumList[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def get_atomicNumList(atomic_symbols):
    atomicNumList = []
    for symbol in atomic_symbols:
        atomicNumList.append(get_atom(symbol))
    return atomicNumList



def xyz2AC(atomicNumList,xyz):

    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i,(xyz[i][0],xyz[i][1],xyz[i][2]))
    mol.AddConformer(conf)

    dMat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms,num_atoms)).astype(int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum())*1.30
        for j in range(i+1,num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum())*1.30
            if dMat[i,j] <= Rcov_i + Rcov_j:
                AC[i,j] = 1
                AC[j,i] = 1

    return AC,mol



def read_xyz_file(filename):
    atomic_symbols  = []
    xyz_coordinates = []

    with open(filename, "r") as file:
        for line_number,line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
                else:
                    charge = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x),float(y),float(z)])

    atomicNumList = get_atomicNumList(atomic_symbols)
    return atomicNumList,xyz_coordinates,charge

#-----
## https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization
def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)

    # ignore stereochemistry for now
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol

def xyz2mol(atomicNumList, xyz_coordinates, charge,charged_fragments,quick):
    AC,mol  = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol


def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    atomicNumList,xyz_coordinates,charge = (filename)
    mol = xyz2mol(atomicNumList, xyz_coordinates, charge, charged_fragments, quick)
    return mol



## champs dataset #############################################################
'''
dsgdb9nsd_000001.xyz

5

C -0.0126981359 1.0858041578 0.0080009958
H 0.0021504160 -0.0060313176 0.0019761204
H 1.0117308433 1.4637511618 0.0002765748
H -0.5408150690 1.4475266138 -0.8766437152
H -0.5238136345 1.4379326443 0.9063972942

'''
def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz=[]
    symbol=[]
    for n in range(num_atom):
        l = line[1+n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]),float(l[2]),float(l[3]),])

    return symbol, xyz


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick   =  True
    charge  = 0
    atom_no = get_atomicNumList(symbol)
    mol     = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol






# check #################################################################
def load_csv(DATA_DIR):

#     DATA_DIR = '/root/share/project/kaggle/2019/champs_scalar/data'

    #structure
    df_structure = pd.read_csv(DATA_DIR + '/structures.csv')

    #coupling
    df_train = pd.read_csv(DATA_DIR + '/train.csv')
    df_test  = pd.read_csv(DATA_DIR + '/test.csv')
    df_test['scalar_coupling_constant']=0
    df_scalar_coupling = pd.concat([df_train,df_test])
    df_scalar_coupling_contribution = pd.read_csv(DATA_DIR + '/scalar_coupling_contributions.csv')
    df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
            how='left', on=['molecule_name','atom_index_0','atom_index_1','atom_index_0','type'])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure       = df_structure.groupby('molecule_name')

    return gb_structure, gb_scalar_coupling


#41169 dsgdb9nsd_042329
def do_one(p):
    i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p

    g = make_graph(molecule_name, gb_structure, gb_scalar_coupling, )
    print(i, g.molecule_name, g.smiles)
    write_pickle_to_file(graph_file,g)


##----
def run_convert_to_graph(graph_dir, csv_dir):
#     graph_dir = '/root/share/project/kaggle/2019/champs_scalar/data/structure/graph1'
    os.makedirs(graph_dir, exist_ok=True)

    gb_structure, gb_scalar_coupling = load_csv(csv_dir)
    molecule_names = list(gb_scalar_coupling.groups.keys())
    molecule_names = np.sort(molecule_names)
    param=[]
    for i, molecule_name in enumerate(molecule_names):

        graph_file = graph_dir + '/%s.pickle'%molecule_name
        p = (i, molecule_name, gb_structure, gb_scalar_coupling, graph_file)

        if i<2000:
            do_one(p)
            
        else:
            param.append(p)

    if 1:
        pool = mp.Pool(processes=16)
        pool.map(do_one, param)
        
        
class ChampsDataset(Dataset):
    def __init__(self, molecule_names, graph_file, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment
        self.graph_file = graph_file
        self.molecule_names = molecule_names

#         self.df = pd.read_csv(DATA_DIR + '/csv/%s.csv'%csv)

        if split is not None:
#             self.id = np.load(DATA_DIR + '/split/%s'%split,allow_pickle=True)
            pass
        else:
            self.id = self.molecule_names

        #zz=0
        #self.dummy_graph = read_pickle_from_file(DATA_DIR + '/structure/graph/dsgdb9nsd_000001.pickle')

    def __str__(self):
            string = ''\
            + '\tmode   = %s\n'%self.mode \
            + '\tsplit  = %s\n'%self.split \
            + '\tcsv    = %s\n'%self.csv \
            + '\tlen    = %d\n'%len(self)

            return string

    def __len__(self):
        return len(self.id)


    def __getitem__(self, index):

        molecule_name = self.id[index]
        graph_file = f'{self.graph_file}/{molecule_name}.pickle'
        graph = read_pickle_from_file(graph_file)
        assert(graph.molecule_name==molecule_name)

        # ##filter only J link
        # if 0:
        #     # 1JHC,     2JHC,     3JHC,     1JHN,     2JHN,     3JHN,     2JHH,     3JHH
        #     mask = np.zeros(len(graph.coupling.type),np.bool)
        #     for t in ['1JHC',     '2JHH']:
        #         mask += (graph.coupling.type == COUPLING_TYPE.index(t))
        #
        #     graph.coupling.id = graph.coupling.id [mask]
        #     graph.coupling.contribution = graph.coupling.contribution [mask]
        #     graph.coupling.index = graph.coupling.index [mask]
        #     graph.coupling.type = graph.coupling.type [mask]
        #     graph.coupling.value = graph.coupling.value [mask]

        if 1:
            atom = System(symbols =graph.axyz[0], positions=graph.axyz[1])
            acsf = ACSF_GENERATOR.create(atom)
            graph.node += [acsf,]


        # if 1:
        #     graph.edge = graph.edge[:-1]

        graph.node = np.concatenate(graph.node,-1)
        graph.edge = np.concatenate(graph.edge,-1)
        return graph
    
    
def null_collate(batch):

    batch_size = len(batch)

    node = []
    edge = []
    edge_index = []
    node_index = []

    coupling_value = []
    coupling_atom_index  = []
    coupling_type_index  = []
    coupling_batch_index = []
    infor = []

    offset = 0
    for b in range(batch_size):
        graph = batch[b]
        #print(graph.molecule_name)

        num_node = len(graph.node)
        node.append(graph.node)
        edge.append(graph.edge)
        edge_index.append(graph.edge_index+offset)
        node_index.append(np.array([b]*num_node))

        num_coupling = len(graph.coupling.value)
        coupling_value.append(graph.coupling.value)
        coupling_atom_index.append(graph.coupling.index+offset)
        coupling_type_index.append (graph.coupling.type)
        coupling_batch_index.append(np.array([b]*num_coupling))

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node
        #print(num_node, len(coupling_batch_index))

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()

    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1),
    ],-1)
    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        
        x = pickle.load(f)
    return x


ACSF_GENERATOR = ACSF(
    species = SYMBOL,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError

class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0

    def __call__(self, time):
        return self.lr

    def __str__(self):
        string = 'NullScheduler\n' \
                + 'lr=%0.5f '%(self.lr)
        return string

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv, self).__init__()

        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim),
            #nn.ReLU(inplace=True),
        )

        self.gru  = nn.GRU(node_dim, node_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(node_dim), 1.0 / math.sqrt(node_dim))


    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()

        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = self.encoder(edge).view(-1,node_dim,node_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)
        message = x_i.view(-1,1,node_dim)@edge
        message = message.view(-1,node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)
        message = F.relu(message +self.bias)

        #2. update: n_j = f(n_j, m_j)
        update = message

        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,node_dim), hidden)
        update = update.view(-1,node_dim)

        return update, hidden

class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star

#message passing
class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=8):
        super(Net, self).__init__()
        self.num_propagate = 8
        self.num_s2s = 6

        self.preprocess = nn.Sequential(
            LinearBn(node_dim, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
        )

        self.propagate = GraphConv(128, edge_dim)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)


        #predict coupling constant
        self.predict = nn.Sequential(
            LinearBn(4*128, 1024),  #node_hidden_dim
            nn.ReLU(inplace=True),
            LinearBn( 1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_target),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index):

        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape

        node   = self.preprocess(node)
        hidden = node.view(1,num_node,-1)

        for i in range(self.num_propagate):
            node, hidden =  self.propagate(node, edge_index, edge, hidden)

        pool = self.set2set(node, node_index)

        #---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = torch.split(coupling_index,1,dim=1)

        pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))

        predict = self.predict(torch.cat([pool,node0,node1],-1))
        predict = torch.gather(predict, 1, coupling_type_index).view(-1)
        return predict
    
from torch_geometric.utils import scatter_
from torch_scatter import *


def do_train(net, schduler, optimizer, train_loader, valid_loader, epochs = 10, verbose=0):
    
    start = timer()    
    start_epoch= 0
    start_iter = 0
    iter_ = start_iter
#     batch_size = 16

    epoch_len = len(train_loader)
    his = []
    for epoch in range(start_epoch, epochs, 1):

        epoch_loss = 0
        for node, edge, edge_index, node_index, coupling_value, coupling_index, infor in train_loader:
            
            # learning rate schduler -------------
            lr = schduler(iter_)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # train  -------------
            net.train()
            node = node.cuda()
            edge = edge.cuda()
            edge_index = edge_index.cuda()
            node_index = node_index.cuda()
            coupling_value = coupling_value.cuda()
            coupling_index = coupling_index.cuda()
            
            optimizer.zero_grad()
            predict = net(node, edge, edge_index, node_index, coupling_index)
            # update loss
            loss = criterion(predict, coupling_value)
            loss.backward()
            optimizer.step()
            
            # print statistics  ------------
            epoch_loss += loss.item()
            iter_ += 1
            
            if verbose > 0:
                print(time_to_str((timer() - start),'min'), f'iter_:{iter_%epoch_len}/{epoch_len}', f'loss {loss.item():.4}', end='',flush=True)
                print('\r',end='',flush=True)
                
        valid_loss = do_valid(net, valid_loader)
        epoch_loss /= len(train_loader)
        
#         d_ = {'epoch':epoch, 'rate':rate,'loss':epoch_loss,'val_loss_1JHC':valid_loss[0],'val_loss_2JHC':valid_loss[1]}
#         d_ = {'val_loss_3JHC':valid_loss[2], 'val_loss_1JHN':valid_loss[3], 'val_loss_2JHN':valid_loss[4], 'val_loss_3JHN':valid_loss[5], 'val_loss_2JHH':valid_loss[6], **d_}
#         d_ = {'val_loss_3JHH':valid_loss[7], 'val_loss':valid_loss[8], 'val_mae':valid_loss[9], 'val_logmae':valid_loss[10], **d_}
#         his.append(d_)
        if verbose > 0:
            time_ = time_to_str((timer() - start),'min')
            print(f'time {time_}, epoch {epoch} rate {rate} loss {epoch_loss:.4} val_loss {valid_loss[10]:.4}')
    return his

def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(valid_loader):

        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
            loss = criterion(predict, coupling_value)

        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += loss.item()
        pass

    valid_loss /= len(valid_loader)

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae   = compute_kaggle_metric( predict, coupling_value, coupling_type,)

    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1
    mae_mean, log_mae_mean = np.sum(mae)/num_target, np.sum(log_mae)/num_target
    valid_loss_detail = log_mae + [valid_loss, mae_mean, log_mae_mean]
    
    return valid_loss_detail

def predict(test_loader):
    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    df_pred = pd.DataFrame()
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor) in enumerate(test_loader):

        #if b==5: break
        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index)
        df_pred_i = pd.DataFrame({'id':infor[0][2], 'scalar_coupling_constant':predict.cpu().detach().numpy() })
        df_pred = pd.concat([df_pred, df_pred_i], axis=0)
    return df_pred


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]

    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def criterion(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss


def criterion_mae(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
#     loss = torch.log(loss)
    return loss    