# JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design
This repository contains code for the paper: [JANUS: Parallel Tempered Genetic Algorithm Guided by Deep Neural Networks for Inverse Molecular Design](https://arxiv.org/abs/2106.04011). 

Originally by: AkshatKumar Nigam, Robert Pollice, Alán Aspuru-Guzik 

Updated by: Ivan Smaliakou, Selvita S.A.


**In case of any issues, found bugs or some clarification requests, please contact me through the Teams or email: ivan.smaliakou@selvita.com**

<img align="center" src="https://raw.githubusercontent.com/aspuru-guzik-group/JANUS/refs/heads/main/aux_files/logo.png"/>


## Prerequsites: 
 - Python 3.10
## Dependencies:
Create a new (recommended!) or use your current one (mind the existing packages, you can muddle while resolving dependency conflicts):
1. `python -m venv venv310`
2. `source ./venv310/bin/activate`
Now we need to install required packages using the following list of commands:
1. pip install --upgrade "numpy<2.0"
2. python3.10 -m pip install rdkit-pypi
3. python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
4. python3.10 -m pip install selfies==2.1.1
4. python3.10 -m pip install joblib
6. python3.10 -m pip install scikit-learn
7. python3.10 -m pip install numpy
8. python3.10 -m pip install pandas

## Prior running a training:
1. Change these functions so that they return values in range `[0, 1]` (that's not a hard constraint, you can use values of any scale, but please stick to the range):
   ```python
   def make_fitness_function(model):
       def fitness_function(smi: str) -> float:
   ```
so that it would return values in range `[0, 1]` (that's not a hard constraint, you can use values of any scale, but please stick to the range).

2. You can change the function `custom_filter` (current filter is not a bad one though). The only requirement is you need to return `True` if the molucule **should remain**, otherwise return `False`.

3. Then change the `params_dict` dictionary if you need to add/modify some parameters. You can also add/modify the params through the `JANUS` class constructor in code, it'll have the same effect as modifying `params_dict`. The params description along with default values and data type is displayed in the table below. Important params are written in **bold font**:

| **Parameter**                   | **Explanation**                                                                                                   | **Default Value**                   | **dtype**   |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------|-------------|
| **`work_dir`**                     | **Directory where the results are saved**                                                                             | None                                | str         |
| **`fitness_function`**            | **User-defined fitness function for given SMILES**                                                                    | None                                | Callable    |
| **`start_population`**            | **File with starting SMILES population (one each line)**                                                                              | None                                | str         |
| `verbose_out`                 | Use verbose output                                                                                                | False                               | bool        |
| **`custom_filter`**               | **Callable filtering function (None defaults to no filtering)**                                                       | None                                | Callable    |
| `alphabet`                    | Alphabet of all available SELFIES for substitution                                                                | None                                | List        |
| `use_gpu`                     | Whether to use CUDA                                                                                               | False                               | bool        |
| `num_workers`                 | Number of workers (change if needed)                                                                              | `multiprocessing.cpu_count()`       | int         |
| **`generations`**                 | **Number of iterations of JANUS to run**                                                                              | 200                                 | int         |
| **`generation_size`**             | **Total number of molecules per generation (exploration and exploitation each get half)**                            | 5000                                | int         |
| `num_exchanges`               | Number of molecules moved from exploration to exploitation per generation                                         | 5                                   | int         |
| **`use_fragments`**               | **Toggle adding fragments from starting population to mutation alphabet from SMILES with some radius (default `radius=3`)**                                     | False                               | bool        |
| `num_sample_frags`            | Number of tokens (atoms) from the SELFIES alphabet used for the mutations                                         | 200                                 | int         |
| `explr_num_random_samples`    | Number of random mutation samples in the exploration population                                                   | 5                                   | int         |
| `explr_num_mutations`         | Number of random mutations per sample in the exploration population                                               | 5                                   | int         |
| `crossover_num_random_samples`| Number of random crossovers                                                                                       | 1                                   | int         |
| `exploit_num_random_samples`  | Number of random mutation samples in the exploitation population                                                  | 400                                 | int         |
| `exploit_num_mutations`       | Number of random mutations per sample in the exploitation population                                              | 400                                 | int         |
| `top_mols`                    | Number of top molecules from exploration to move to exploitation                                                  | 1                                   | int         |
| `max_same_best`               | Max number of consecutively generated same best SMILES in `RESULTS_DIR/generation_all_best.txt`                  | 5                                   | int         |


## Training JANUS:
As everything is set up, we can run the generation process.
1. `python -m janus_run.py`

## Anylizing the results
All results from running JANUS will be stored in specified `work_dir`. 
The following files will be created: 
1. fitness_explore.txt: 
   Fitness values for all molecules from the exploration component of JANUS.    
2. fitness_local_search.txt: 
   Fitness values for all molecules from the exploitation component of JANUS. 
3. generation_all_best.txt: 
   Smiles and fitness value for the best molecule encountered in every generation (iteration). 
4. init_mols.txt: 
   List of molecules used to initialte JANUS. 
5. population_explore.txt: 
   SMILES for all molecules from the exploration component of JANUS. 
6. population_local_search.txt: 
   SMILES for all molecules from the exploitation component of JANUS. 
7. hparams.json:
   Hyperparameters used for initializing JANUS.


**As a rule, `population_explore.txt` should always contain more diverse set with higher fitness value, while `population_local_search.txt` would contain molecules with lower fitness value, but the ones that are closer to the start population.**


Then you can use `janus_evaluate.ipynb` for some basic data analysis. The Jupyter Notebook has already the Markdown cells to guide you through the results analysis.

## Major changes:

- Support the use of any version of SELFIES (please check your installation).
- Multiprocessing is now disabled in calculating fitness function value, mutation and crossover processes due to the high degree of instability;
- Classification for high/low fitness is replaced with calculating the fitness value itself in molecules overflow filtering; 
- Early stopping conducted if the best molecule doesn't change for `max_same_best` consequent generation iterations;

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
