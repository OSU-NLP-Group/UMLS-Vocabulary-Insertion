"""
Script to evaluate kNN retrieval using either pre-computed vectors or a BERT model available in
the Huggingface model library or saved locally.

Inputs

- UMLS Directory: Path to the 'META' directory which stores UMLS tables.

- UMLS Version: Version of UMLS which contains all the AUIs referred to by the AUI sets.

- Original AUI Set: Set of AUIs to be used as the original ontology (the ontology to which new
AUIs will be attached).

- New AUI Set: Set of AUIs to be attached to "original" ontology.

- K: Number of nearest neighbors to extract from original ontology for
each new term.

- Output Directory: Path to the directory used for saving output files.

- Vector Dictionary (Optional): Python dictionary mapping AUIs or Strings to numpy vectors.

- BERT Model (Optional): If no vector dictionary, BERT model is necessary to extract representations

- TODO: New Term Set (Optional): If terms are not AUIs they can still be linked to the original ontology using only strings.

Outputs

- Set of K nearest neighbors from original ontology for each new AUI or term.
- AUI Based Recall @ 1,5,10,50,100,200,1000,2000
- CUI Based Recall @ 1,5,10,50,100,200,1000,2000
"""

import os
import sys
from UMLS import UMLS


def main():
    umls_dir = sys.argv[1]
    aui_dir = sys.argv[2]
    umls_version = sys.argv[3]
    original_auis_filename = sys.argv[4]
    new_auis_filename = sys.argv[5]
    k = int(sys.argv[6])
    output_dir = sys.argv[7]
    num_retriever_names = int(sys.argv[8])
    retriever_names = sys.argv[9:9 + num_retriever_names]

    umls = UMLS(umls_directory=umls_dir, 
                aui_directory=umls_version,
                version=umls_version
               )
    umls.augment_umls(original_auis_filename,
                      new_auis_filename,
                      output_dir,
                      retriever_names,
                      k,
                      None,
                      candidates_to_classify=None
                      )

if __name__ == "__main__":
    main()