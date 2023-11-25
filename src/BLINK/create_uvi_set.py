import os
import pickle
import numpy as np
import torch
import pandas as pd
from glob import glob
import json
from tqdm import tqdm
import ipdb
from transformers import AutoTokenizer, PreTrainedTokenizer, BertTokenizerFast
import faiss


from sklearn.metrics import recall_score,f1_score,accuracy_score


import os
import sys
sys.path.insert(0, os.path.abspath('../'))

from UMLS import UMLS
from RetrievalModule import RetrievalModule

umls_version = sys.argv[1]
candidate_directory_num = sys.argv[2]

mrconso_file = '../../data/umls_versions/{}-ACTIVE/META_DL/MRCONSO_MASTER.RRF'.format(umls_version)
mrconso_rba_link = '../../data/umls_versions/{}-ACTIVE/META/CHOSEN_AUIS.txt'.format(umls_version)
new_aui_set_file = '../../data/insertion_sets/{}_insertion.txt'.format(umls_version)
rba_filename = '../../data/umls_versions/{}-ACTIVE/RBA_V2/AUI_COLOR.PICKLE'.format(umls_version)
sort_filename = '../../output/{}/cambridgeltl_SapBERT-from-PubMedBERT-fulltext_candidates.p'.format(str(candidate_directory_num))


umls = UMLS(mrconso_file)
umls.get_new_aui_set(new_aui_set_file)

candidates = pickle.load(open(sort_filename,'rb'))

mrconso = open(mrconso_file,'r').readlines()
chosen_auis = pd.read_csv(mrconso_rba_link,sep='|',header=None)
    
mrconso_dict = {}

for idn, aui in tqdm(zip(chosen_auis[0], chosen_auis[1])):    
    mrconso_dict[int(idn)] = aui
    
colors = pickle.load(open(rba_filename,'rb'))

color2aui = {}
aui2color = {}

for idn,color in tqdm(colors.items()):
    
    color_set = color2aui.get(color,set())
    aui = mrconso_dict[idn]
    
    color_set.add(aui)

    color2aui[color] = color_set
    aui2color[aui] = color

rba_predicted_synonyms = {}

for new_aui in tqdm(umls.new_auis):
    
    color = aui2color[new_aui]
    predicted_auis = color2aui[color]
    
    filtered_preds = set()
    
    for pred in predicted_auis:
        if pred != aui and pred not in umls.new_auis:
            filtered_preds.add(pred)
            
    rba_predicted_synonyms[new_aui] = [umls.aui2cui[aui] for aui in filtered_preds]

if '2020AB' in mrconso_file:
    dataset_splits = pickle.load(open('../../data/splits/2020AB_training_dev_test_split.p','rb'))
    dataset_splits = dataset_splits[['auis','split']].set_index('auis').to_dict()
    dataset_splits = dataset_splits['split']
else:
    dataset_splits = None

old_cuis = set()

for aui in tqdm(umls.original_auis):
    
    old_cuis.add(umls.aui2cui[aui])

df = []

for aui,cand_auis in tqdm(candidates.items()):
    query_str = umls.aui2str[aui]
    cand_auis, cand_dists = cand_auis
    
    true_cui = umls.aui2cui[aui]
    cand_cuis = [umls.aui2cui[aui_cand] for aui_cand in cand_auis]
    cand_strs = [umls.aui2str[aui_cand] for aui_cand in cand_auis]

    null_or_cui = true_cui not in old_cuis
    
    rba_cands = rba_predicted_synonyms[aui]
    
    if dataset_splits is not None:
        split = dataset_splits[aui]
    else:
        split = 'dev'
        
    sem_group = umls.cui2sg[true_cui]
    
    df.append((aui, query_str, true_cui, sem_group, cand_auis, cand_strs, cand_dists, cand_cuis, null_or_cui, rba_cands, split))
    
df = pd.DataFrame(df,columns=['aui','query_str', 'true_cui', 'sem_group', 'cand_auis', 'cand_strs', 'cand_dists', 'cand_cuis', 'null_or_cui', 'rba_cands', 'split'])


## If split is not defined by file as with 2020AB, stratify by semantic group into 50/50 dev-test split.

if '2020AB' not in mrconso_file:
    cui_sem_group_cui = df[['true_cui','sem_group']].drop_duplicates()
    validation = []
    testing = []

    test = 0.50

    for i,g in cui_sem_group_cui.groupby('sem_group'):

        perm_g = g.sample(len(g),random_state=np.random.RandomState(42)).true_cui.values

        validation.extend(perm_g[:len(g) - int(len(g)*(test))])
        testing.extend(perm_g[len(g) - int(len(g)*test):])

        assert(validation[-1] != testing[0])

    validation = set(validation)
    testing = set(testing)

    splits = []

    for cui in df.true_cui:

        if cui in testing:
            splits.append('test')
        else:
            splits.append('dev')


    df['split'] = splits

## Creating Re-Ranking Dataset

print('Creating a Re-Ranking Dataset')

directory = 'models'

for add_rba_info in [True, False]:
    max_len = 64
    num_candidates = 50
    subsets = {'train':None,'valid':None,'test':None}

    pubmed_bert_tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

    if subsets['train'] is None:
        train_size = 'all'
    else:
        train_size = subsets['train']
        
    data_name = 'uva_ins_{}_max_len_{}_num_cands_{}_train_size_{}_sorted_cuis'.format(umls_version, max_len, num_candidates, train_size)

    if add_rba_info:
        data_name += '_rba_info'

    if not(os.path.exists('{}'.format(directory))):
        os.makedirs('{}'.format(directory))

    if not(os.path.exists('{}/{}'.format(directory, data_name))):
        os.makedirs('{}/{}'.format(directory, data_name))

    split_datasets = {}
    split_datasets_strings = {}
    final_datasets = {}

    for split_name,g in df.groupby('split'):
        
        print((add_rba_info, split_name))
        
        if split_name == 'val' or split_name == 'dev':
            split_name = 'valid'

        subset_size = subsets[split_name]
        
        if subset_size is not None:
            subset = []

            for j,sem_group in g.groupby('sem_group'):
                subset.append(sem_group.sample(int(subset_size*len(sem_group)/len(g)),random_state=np.random.RandomState(42)))

            g = pd.concat(subset)

        queries = []
        candidates = []        
        
        query_token_ids = []
        candidate_token_ids = []

        overall_candidate_origin = []
        overall_candidate_cuis = []
        overall_candidate_auis = []

        labels = []

        kept_token_ids = []
        ignored_rows = 0
        padded_rows = 0

        for j, row in tqdm(g.iterrows(),total=len(g)):
            query_aui = row.aui
            true_cui = row.true_cui
            null_cui = row.null_or_cui

            rba_cuis = list(np.sort(list(set(row.rba_cands)),kind='stable'))

            sorted_cuis = row.cand_cuis
            sorted_auis = row.cand_auis

            query_str = umls.aui2str[query_aui]

            if len(rba_cuis) == 0 and add_rba_info:
                query_str += ' (No Preferred Candidate)'

            encoded_query = pubmed_bert_tokenizer.encode(query_str, 
                                                         max_length=max_len,
                                                         padding='max_length',
                                                         truncation=True)

            label = None
            candidate_strings_tuple = []

            seen_cuis = set()

            #Setting RBA CUIs before sorted CUIs
            all_candidate_cuis = rba_cuis + sorted_cuis
            cui_origins = ['RBA' for _ in rba_cuis] + ['SORT_ALG' for _ in sorted_cuis]

            #Adding RBA AUIs
            rba_auis = []
            for cui in rba_cuis:
                rba_auis.extend(umls.cui2auis[cui])

            all_candidate_auis = sorted_auis + rba_auis

            for cand_ind, candidate in enumerate(all_candidate_cuis):
                
                cand_origin = cui_origins[cand_ind]

                if candidate not in seen_cuis:
                    if true_cui == candidate:
                        label = len(candidate_strings_tuple)

                    #Getting Closest AUI String
                    for aui in all_candidate_auis:
                        if candidate == umls.aui2cui[aui]:
                            chosen_aui = aui
                            break

                    #Encoding CUI as Closest AUI
                    chosen_candidate_str = umls.aui2str[chosen_aui]

                    candidate_str = chosen_candidate_str

                    if candidate in rba_cuis and add_rba_info:
                        candidate_str = candidate_str + ' (Preferred)'

                    candidate_strings_tuple.append((candidate, chosen_aui, candidate_str, cand_origin))

                    if len(candidate_strings_tuple) == num_candidates - 1:
                        break

                    seen_cuis.add(candidate)

            if null_cui:
                assert label is None
                label = len(candidate_strings_tuple)

            candidate_strings_tuple.append(('NULL','NULL','NULL','NULL'))

            if len(candidate_strings_tuple) < num_candidates - 1:
                padded_rows += num_candidates - len(candidate_strings_tuple)

            while len(candidate_strings_tuple) != num_candidates:
                candidate_strings_tuple.append(('[PAD]','[PAD]','[PAD]','[PAD]'))

            assert len(candidate_strings_tuple) == num_candidates

            candidate_strings = [t[2] for t in candidate_strings_tuple]
            encoded_candidate_token_ids = pubmed_bert_tokenizer.batch_encode_plus(candidate_strings, 
                                                             max_length=max_len,
                                                             padding='max_length',
                                                             truncation=True)['input_ids']

            if label is None:
                ignored_rows += 1
            else:
                queries.append(query_str)
                candidates.append(candidate_strings)
                overall_candidate_cuis.append([t[0] for t in candidate_strings_tuple])
                overall_candidate_auis.append([t[1] for t in candidate_strings_tuple])
                overall_candidate_origin.append([t[3] for t in candidate_strings_tuple])
                
                query_token_ids.append(encoded_query)
                candidate_token_ids.append(encoded_candidate_token_ids)

                labels.append(label)
                kept_token_ids.append(j)

        split_datasets[split_name] = (query_token_ids, candidate_token_ids, labels)
        split_datasets_strings[split_name] = (queries,candidates,labels)
        
        g = g.loc[kept_token_ids]
        g['overall_candidate_strings'] = candidates
        g['overall_candidate_auis'] = overall_candidate_auis
        g['overall_candidate_cuis'] = overall_candidate_cuis
        g['overall_candidate_origin'] = overall_candidate_origin
        
        final_datasets[split_name] = g

        print('Ignored {} Queries. Mean Padded Query {}'.format(ignored_rows, padded_rows/len(queries)))


    pickle.dump(final_datasets,open('{}/{}/cui_based_datasets.p'.format(directory, data_name),'wb'))

    if not(os.path.exists('{}/{}/{}'.format(directory, data_name, 'top{}_candidates'.format(num_candidates)))):
        os.makedirs('{}/{}/{}'.format(directory, data_name, 'top{}_candidates'.format(num_candidates)))

    for k, dataset in split_datasets.items():
        split_data = {}
        print(np.unique(dataset[2]))
        print(len(dataset[2]))

        split_data['context_vecs'] = torch.Tensor(dataset[0]).type(torch.LongTensor)
        split_data['candidate_vecs'] = torch.Tensor(dataset[1]).type(torch.LongTensor)
        split_data['labels'] = torch.Tensor(dataset[2]).type(torch.LongTensor)

        torch.save(split_data,'{}/{}/top{}_candidates/{}.t7'.format(directory, data_name, num_candidates, k))

