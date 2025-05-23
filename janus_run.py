from janus import JANUS, utils
from rdkit import Chem, RDLogger
from alex_rings_filters.rings import filter_rings
import pandas as pd
import numpy as np
import selfies
import joblib
import torch
import re
from rdkit.Chem import AllChem, RDConfig, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogEntry, RunFilterCatalog, SmartsMatcher

RDLogger.DisableLog("rdApp.*")

from rdkit.DataStructs import TanimotoSimilarity

FILTERS_PATH = './filters/rd_filters.csv'
ACTIVITY_SCORE_MULT = 2.0

rd_filters = pd.read_csv(FILTERS_PATH)
chembl_rings = pd.read_pickle('./rings_filters/chembl_rings.pkl')
filters = FilterCatalog()
for i, row in rd_filters.iterrows():
    matcher = SmartsMatcher('', row['smarts'])
    entry = FilterCatalogEntry(row['description'], matcher)
    filters.AddEntry(entry)

print(f'Cuda is available: {torch.cuda.is_available()}')

def get_atom_chars(smi):
    atoms_chars=[]
    mol = Chem.MolFromSmiles(smi,sanitize=False)
    for a in mol.GetAtoms():
        atom=Chem.RWMol()
        atom.AddAtom(a)
        atoms_chars.append(Chem.MolToSmiles(atom))
    return atoms_chars

def has_radicals(mol):
    for atom in mol.GetAtoms():
      if atom.GetNumRadicalElectrons() > 0:
        return True
    return False

def is_neutral(mol):
  for atom in mol.GetAtoms():
    charge = atom.GetFormalCharge()
    if charge != 0:
      return False  # Atom is charged
  return True  # All atoms are neutral

def is_organic(mol):
    allowed_elements = {'H', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_elements:
            return False
    return True
patterns = [Chem.MolFromSmarts('[#6,O]-[P](=[O,N])(=[O,N])-[O,N]'), Chem.MolFromSmarts('[#6]-[SX2]-[#6]'), Chem.MolFromSmarts('[#6]-[SX3](=O)-[#6]'),
             Chem.MolFromSmarts('[#6]-[S](=[O])(=[O])-N'), Chem.MolFromSmarts('[sX2]'), Chem.MolFromSmarts('[sX3]'), Chem.MolFromSmarts('[P]'),
               Chem.MolFromSmarts('[r3]=[r3]'), Chem.MolFromSmarts('[r4]=[r4]'), Chem.MolFromSmarts('[r]#[r]'), Chem.MolFromSmarts('[*]=[#6]=[*]'),
               Chem.MolFromSmarts('[#6]-1=[#6]-[#6]=[#6]-[#6]-1')]

def filter_smarts(mol):
    for pattern in patterns:
        # Check if SMARTS pattern matches the molecule
        if mol.HasSubstructMatch(pattern):
            return True
    return False

def make_fitness_function(model):
  def fitness_function(smi: str) -> float:
      """ User-defined function that takes in individual smiles 
      and outputs a fitness value.
      """
      # logP fitness
      # return Descriptors.MolLogP(Chem.MolFromSmiles(smi))
      #return calculate_tanimoto_similarity(smi, "COc1c2c(cc(c1N3C[C@@H]4CCCN[C@@H]4C3)F)c(=O)c(cn2C5CC5)C(=O)O")
      mol = Chem.MolFromSmiles(smi, sanitize=True)
      X_fp = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048).ToList()])
      activity_score = (model.predict(X_fp).item() - 4) / 7 # scale to ~ [0, 1]
      qed = QED.qed(mol)
      sa_score = sascorer.calculateScore (mol) / 10 # scale to [0, 1]
      # consider filters in the fitness fucntion
      filter_out = RunFilterCatalog(filters, [smi], numThreads=40)
      # emphasize the activity score by mul by the weight; 
      # add top 50 diverse molecules to the start population  -picker.LazyBitVectorPick;
      # run 2 times: with active molecules included and w/o;
      
      fitness_val = (ACTIVITY_SCORE_MULT * activity_score + qed - sa_score) / 4 # normalize once again to [0, 1].
      if len(filter_out[0]) > 0:
        fitness_val *= 0.5
      return fitness_val
  return fitness_function
    

def custom_filter(smi: str):
    """ Function that takes in a smile and returns a boolean.
    True indicates the smiles PASSES the filter.
    """
    # smiles length filter
    # remove the upper boundary and add a lower boundary ~ > 5 chars;
    chars = get_atom_chars(smi)
    mol = Chem.MolFromSmiles(smi)
    for char in chars: # remove boron and check phosphorus and sulfur
       if re.match(r'\[?[Bb][+-]?\]?', char) or ((re.match(r'\[?[Pp][+-]?\]?', char) or re.match(r'\[?[Ss][+-]?\]?', char)) and not filter_smarts(mol)):
          return False
    return len(smi) >= 6 and not has_radicals(mol) and is_neutral(mol) and is_organic(mol) and filter_rings(smi=smi, chembl_rings=chembl_rings, count_threshold=100)
    
torch.multiprocessing.freeze_support()

# all parameters to be set, below are defaults
params_dict = {
    # Number of iterations that JANUS runs for
    "generations": 100,
    # The number of molecules for which fitness calculations are done, 
    # exploration and exploitation each have their own population
    "generation_size": 1000,
       
    # Number of molecules that are exchanged between the exploration and exploitation
    "num_exchanges": 5,

    # Callable filtering function (None defaults to no filtering)
    "custom_filter": custom_filter,

    # Fragments from starting population used to extend alphabet for mutations
    "use_fragments": False,

    # An option to use a random sampling when filtering the molecule overflow
    "use_random": False,

    # max number of same best molecules in a generation
    "max_same_best": 5,
}

 # Set your SELFIES constraints (below used for manuscript)
default_constraints = selfies.get_semantic_constraints()
# new_constraints = default_constraints
# new_constraints['S'] = 2
# new_constraints['P'] = 3
# selfies.set_semantic_constraints(new_constraints)  # update constraints
model = joblib.load("./activity_prediction/svr_model.pkl")
# Create JANUS object.
agent = JANUS(
    work_dir = 'RESULTS_chembl_best_act_mix',                                   # where the results are saved
    fitness_function = make_fitness_function(model),                    # user-defined fitness for given smiles
    start_population = "./tests/DATA/chembl_best_act_mix.txt",   # file with starting smiles population
    alphabet=list(selfies.get_semantic_robust_alphabet()),
    num_workers=1,
    top_mols=10,  # number of top molecules to keep in each generation
    exploit_num_random_samples=10,
    **params_dict
)
agent.run()