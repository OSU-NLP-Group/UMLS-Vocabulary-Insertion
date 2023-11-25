import _pickle as pickle
import json

import pandas as pd
from tqdm import tqdm
import time
from UMLS import *
from RetrievalModule import *
from utils import *

import ipdb
from glob import glob

class RetrievalPipeline:
    """
    Class designed to run several retrieval modules.
    """

    def __init__(self,
                 new_auis_filename,
                 ontology,
                 output_dir,
                 maximum_candidates_per_retriever,
                 retriever_names,
                 distance='COSINE'
                 ):

        self.ontology = ontology
        self.maximum_candidates_per_retriever = maximum_candidates_per_retriever
        self.retriever_names = retriever_names        
        self.distance = distance

        self.retrieved_candidates = {}

        self.relevant_auis = self.ontology.relevant_auis
        
        self.original_auis, self.new_auis = self.ontology.get_new_aui_set(new_auis_filename)

        self.new_aui_synonyms = []

        if self.ontology.original_only_cui2auis is None:
            print('Populating original only synonyms before evaluation.')
            self.ontology.get_original_ontology_synonyms(self.original_auis)

        self.load_retrievers()
        self.retrieval_done = False

        # Make Unique Directory for this retrieval procedure
        configs = {'MRCONSO Directory': ontology.mrconso_file,
                   'Retriever Names': retriever_names,
                   'New AUI Filename': new_auis_filename,
                   'Maximum Candidates per Retriever': maximum_candidates_per_retriever,
                   'Distance Metric':distance
                  }
        retrieval_directories = glob(output_dir + '/*')

        new_directory_num = len(retrieval_directories)

        for dir in retrieval_directories:
            prev_config = json.load(open('{}/config.json'.format(dir),'r'))

            if equivalent_dict(prev_config, configs):
                if os.path.exists('{}/retrieval_done.json'.format(dir)):
                    print('Configuration Already Done and Saved.')
                    self.retrieval_done = True
                else:
                    print('Previous Run Stopped. Running Again.')

                new_directory_num = dir.split('/')[-1]

        self.output_dir = output_dir + '/{}'.format(new_directory_num)

        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
            json.dump(configs, open(self.output_dir + '/config.json', 'w'))

    def load_retrievers(self):
        self.retrievers = []

        for retriever_name in self.retriever_names:
            self.retrievers.append(RetrievalModule(retriever_name,
                                                   self))

    def combine_candidates(self, new_candidate_dict, add_on_top=False):

        for new_aui in self.new_auis:

            current_candidates = self.retrieved_candidates.get(new_aui, [])
            new_candidates = new_candidate_dict[new_aui]

            # Add new candidates before or after previous ones
            if add_on_top:
                current_candidates = new_candidates + current_candidates
            else:
                current_candidates = current_candidates + new_candidates

            self.retrieved_candidates[new_aui] = current_candidates

    def run_retrievers(self,
                       exclude=[]):

        if self.retrieval_done:
            print('Retrieval Done.')
        else:
            for ret_name, ret in zip(self.retriever_names, self.retrievers):

                if ret_name not in exclude:
                    print('Retrieving {} candidates.'.format(ret_name))
                    new_candidate_dict = ret.retrieve()
                    self.eval_and_save_candidates(new_candidate_dict, ret_name)
#                     self.combine_candidates(new_candidate_dict, ret.add_on_top)

#             self.eval_and_save_candidates(self.retrieved_candidates, 'full_pipeline')

            json.dump({'DONE':True}, open(self.output_dir + '/retrieval_done.json', 'w'))

    def evaluate_candidate_retrieval(self,
                                     candidate_dict,
                                     mode,
                                     recall_at=[1, 5, 10, 50, 100, 200, 500, 1000, 2000]):

        self.new_aui_synonyms = []

        new_auis = []
        recall_array = []

        for new_aui, candidate_info in tqdm(candidate_dict.items()):
            candidates, candidate_dists = candidate_info
            
            new_auis.append(new_aui)

            cui = self.ontology.aui2cui[new_aui]
            true_syn = set(self.ontology.original_only_cui2auis.get(cui, []))
            self.new_aui_synonyms.append(true_syn)

            if len(true_syn) > 0:
                if mode == 'CUI':
                    true_syn = {cui}
                    
                    new_candidates = []
                    candidate_set = set()
                    for aui in candidates:
                        cui = self.ontology.aui2cui[aui]
                        if cui not in candidate_set:
                            new_candidates.append(cui)
                            candidate_set.add(cui)
                    
                    candidates = new_candidates

                recalls = []

                for n in recall_at:
                    topn = set(candidates[:n])
                    true_pos = topn.intersection(true_syn)

                    # Number of true positives in first n over the number of possible positive candidates
                    # (if n is less than the number of true synonyms it is impossible to correctly recall all of them)
                    recall_at_n = len(true_pos) / len(true_syn)
                    recalls.append(recall_at_n)

                recall_array.append(recalls)
            else:
                recalls = []

                recall_array.append(recalls)

        return pd.DataFrame(recall_array, index=new_auis, columns=['R@{}'.format(n) for n in recall_at])

    def eval_and_save_candidates(self, candidate_dict, ret_name):
        ret_name = ret_name.replace('/', '_')  # In case using filename

        aui_recall = self.evaluate_candidate_retrieval(candidate_dict, mode='AUI')
        cui_recall = self.evaluate_candidate_retrieval(candidate_dict, mode='CUI')

        aui_recall.to_csv('{}/{}_aui_recall_complete.csv'.format(self.output_dir, ret_name))
        cui_recall.to_csv('{}/{}_cui_recall_complete.csv'.format(self.output_dir, ret_name))

        aui_mean_row = aui_recall.agg('mean')
        cui_mean_row = cui_recall.mean()
        metrics = pd.DataFrame([aui_mean_row, cui_mean_row])
        metrics.index = ['{}_AUI_metrics'.format(ret_name), '{}_CUI_metrics'.format(ret_name)]
        metrics.to_csv('{}/{}_recall_summary.csv'.format(self.output_dir, ret_name))

        pickle.dump(candidate_dict, open('{}/{}_candidates.p'.format(self.output_dir, ret_name),'wb'))