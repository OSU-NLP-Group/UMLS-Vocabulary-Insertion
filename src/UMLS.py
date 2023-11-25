import _pickle as pickle
import pandas as pd
from tqdm import tqdm
import time
from RetrievalPipeline import RetrievalPipeline
import numpy as np
import ipdb

class UMLS:
    """
    Class designed to hold and manage the UMLS Ontology.
    """

    def __init__(self, mrconso_file):

        self.mrconso_file = mrconso_file

        # Load all UMLS Info Necessary
        self.aui_info = []

        # CUI-AUI Mapping
        self.cui2auis = {}
        self.aui2cui = {}
        self.cui_aui = []

        # AUI-Source CUI Mapping
        self.aui2scui = {}
        self.scui2auis = {}

        # Preferred Name Mapping
        self.cui2preferred = {}

        # AUI-String Mapping
        self.aui2str = {}
        self.str2aui = {}

        # Semantic Type & Group Mapping
        self.aui2sg = {}
        self.cui_sg = []
        self.cui2sg = {}
        self.semtype2sg = {}

        # AUI-LUI Mapping
        self.aui2lui = {}
        self.lui2auis = {}

        self.original_only_lui2auis = None
        self.original_only_scui2auis = None
        self.original_only_cui2auis = None

        self.relevant_auis = set()

        # Loading UMLS Info from File
        self.raw_load_umls(mrconso_file)
        self.create_mappings()

    def raw_load_umls(self, 
                      mrconso_file):

        print('Loading Raw MRCONSO Lines')
        # Download all MRCONSO
        with open(mrconso_file, 'r') as fp:
            pbar = tqdm(total=15000000)
            line = fp.readline()
            
            while line:
                line = line.split('|')
                cui = line[0]
                aui = line[1]
                string = line[2]
                sg = line[3].strip()

                self.cui2sg[cui] = sg
                
                self.aui_info.append((aui, cui, string, sg))
                self.relevant_auis.add(aui)
                    
                line = fp.readline()
                pbar.update(1)

    
    def get_new_aui_set(self, new_auis_filename):
        
        # Load Original and New AUIs (Synonyms obtained only from AUIs in these 2 sets)
        new_auis = open(new_auis_filename, 'r').readlines()
        new_auis = set([a.strip() for a in new_auis])

        # Keep only AUIs that are present in UMLS Version (Print Number of AUIs missing)
        new_auis_in_ontology = new_auis.intersection(set(self.aui2cui.keys()))

        print('{} AUIs are not present in UMLS Version and so are ignored.'.format(len(new_auis) - len(new_auis_in_ontology)))
        new_auis = new_auis_in_ontology

        original_auis = self.relevant_auis.difference(new_auis)

        self.original_auis = original_auis
        self.new_auis = new_auis

        return original_auis, new_auis

    def create_mappings(self):
        print('Creating mappings between concept IDs for easy access.')

        for tup in tqdm(self.aui_info):
            current_time = time.time()

            aui = tup[0]
            cui = tup[1]
            string = tup[2]

            sg = self.cui2sg[cui]

            #Preferred is no longer true preferred
            self.cui2preferred[cui] = string

            self.aui2str[aui] = string
            self.aui2cui[aui] = cui
            self.aui2sg[aui] = sg

            auis = self.str2aui.get(string, [])
            auis.append(aui)
            self.str2aui[string] = auis

            self.cui_sg.append((cui, sg))
            self.cui_aui.append((cui, aui))

            #Only Obtain Synonyms from AUIs defined
            if aui in self.relevant_auis:
                auis = self.cui2auis.get(cui, [])
                auis.append(aui)
                self.cui2auis[cui] = auis

                if (time.time() - current_time) > 5:
                    print(tup)

    def get_original_ontology_synonyms(self, original_auis):
        """
        Build CUI to AUI set, SCUI to AUI and LUI to AUI set mappings
        which only contain AUIs from the "original" ontology.
        """

        self.original_only_cui2auis = {}

        for aui in tqdm(original_auis):
            cui = self.aui2cui[aui]

            auis = self.original_only_cui2auis.get(cui, [])
            auis.append(aui)
            self.original_only_cui2auis[cui] = auis

    def augment_umls(self,
                     new_aui_filename,
                     output_dir,
                     retriever_names,
                     maximum_candidates,
                     classifier_name,
                     distance='L2',
                     candidates_to_classify=100,
                     add_gold_candidates=False,
                     dev_perc=0.0,
                     test_perc=0.0):
        """
        This method enables new terms to be introduced automatically into the UMLS Ontology using only their strings
        and any synonymy information available from its source ontology.

        Args:
            original_aui_filename: Filename with a list of AUIs to use as original or "base" ontology.
            (Pickle file w/ iterable)
            new_aui_filename: Filename with a list of AUIs that are to be introduced into the original or "base"
            ontology. (Pickle file w/ iterable)
            retriever_names: List of names for all retriever systems to be used in producing candidate AUIs from the
            original ontology for each new AUIs.
            maximum_candidates: Maximum # candidates per retrieval method
            classifier_name: Name of classification system (For now the location of a fine-tuned PLM model)
            candidates_to_classify: # of candidates to pass through classification system
            add_gold_candidates: Whether to add synonyms to retrieved candidates (Defaults to no for inference run)
            dev_perc: Percentage of total concepts to introduce into the development set
            test_perc: Percentage of total concepts to introduce into the test set

        Returns:
            predicted_synonymy: Dictionary linking each new AUI to a list of AUIs from the original ontology
            which are predicted to be synonymous.
            evalution_metrics:
                - Recall @ Various N's for each retriever
                - F1, Precision and Recall for final classifier
        """

        # Create Retrieval Pipeline
        self.retriever_pipeline = RetrievalPipeline(new_aui_filename,
                                                    self,
                                                    output_dir,
                                                    maximum_candidates,
                                                    retriever_names
                                                   )

        # Run Candidate Generation (Retrieval)
        self.retriever_pipeline.run_retrievers()
