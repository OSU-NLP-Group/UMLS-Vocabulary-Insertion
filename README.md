# UMLS Vocabulary Insertion

This repository provides code to run UMLS vocabulary insertion from publicly available UMLS files. We provide an implementation for the best models in our Findings in EMNLP 2023 paper "Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion". 

## Installation

Run the following commands to create a conda environment with the required packages. 

```
conda create -n uvi python=3.9 pip
conda activate uvi
pip install -r requirements.txt
```

## Data

Please download the data directory from this [link](https://drive.google.com/drive/folders/1tUQtzttOlDiWMCweCHlATwFWqypYDuFj?usp=sharing) and use it to replace the existing `data` directory.

Due to UMLS license, we are unable to distribute the complete UMLS files. To run the UMLS vocabulary insertion, please obtain a UMLS license by following the following [instructions](https://www.nlm.nih.gov/databases/umls.html).

To reproduce the results in our paper on the 2020AB to 2022AB UMLS insertion sets, download the full UMLS files for each version [here](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsarchives04.html). After downloading them, place the files under the `META` directory (only MRCONSO.RRF and MRSTY.RRF are actually needed) under the existing META directories `data/umls_versions/20*-ACTIVE/META`.

After placing all files in the appropriate places, run the script `data/create_intermediate_umls_files.py` to create the intermediate files necessary for further experiments. All other files necessary to reproduce our results on these 5 UMLS updates are included in this repository. 

Please reach out to [jimenezgutierrez.1@osu.edu](mailto:jimenezgutierrez.1@osu.edu) for additional files (mainly for the RBA system) which would be necessary to run UMLS vocabulary insertion in other versions.


## Reproducing Experiments


### Candidate Generation

In order to run candidate generation for all UMLS insertion sets in the paper (2020AB to 2022AB) using the best performing SapBERT model, run the following script under the `src` directory:

```
CUDA_VISIBLE_DEVICES=0 bash umls_candidate_generation.sh
```

Once this candidate generation has finished, each insertion set will have a directory under the `output` directory with ranked candidates for each new atom and embedding model.

### Re-Ranking Data Processing

To train the re-ranking module, we first build a training set from the UMLS 2020AB insertion set and the best distance based model (SapBERT) using the following commands:

```
cd BLINK
bash create_uvi_sets.sh
```

This script runs the generation for all 5 insertion sets tested. After it runs, the top candidates from the (RBA + SapBERT) baseline should have been re-formatted in accordance with the BLINK codebase. Two versions of each dataset will be created, one with RBA signal and one without, as described in our paper. 

### Re-Ranking Training

After creating the appropriate training data from the 2020AB training set we run the following commands to train the re-ranker. 

```
bash train.sh
```

Be advised that the size of the datasets makes this process quite time consuming. In case you want to debug the code before a full scale run, add the `--debug` argument to the commands in `train.sh`.

### Re-Ranking Inference

Once the re-ranking models with and without RBA signal are trained, we will use them on the other insertion sets we created earlier. To do this, we run the following commands:

```
best_no_rba_model_name=models/uva_ins_2020AB_max_len_64_num_cands_50_train_size_all_sorted_cuis/epoch_0/pytorch_model.bin
bash generalization_run.sh $best_no_rba_model_name

best_rba_model_name=models/uva_ins_2020AB_max_len_64_num_cands_50_train_size_all_sorted_cuis_rba_info/epoch_0/pytorch_model.bin
bash generalization_run_rba_info.sh $best_rba_model_name
```

The `best_no_rba_model_name` and `best_rba_model_name` variables should be set to the `.bin` file of the models with the highest validation set accuracy in the 2020AB insertion set. The above models are only example filenames.

### Overall Evaluation

Finally, we can check the overall performance of the baselines and re-ranking model on different versions using the Jupyter notebook titled ```eval_all_uvi_methods.ipynb```.  Setting the current version variable is all that would be necessary before running the entire notebook.