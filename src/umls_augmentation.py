from UMLS import *
import sys

def main():
    mrconso_file = sys.argv[1]
    new_auis_filename = sys.argv[2]
    output_dir= sys.argv[3]
    
    retriever_names=['cambridgeltl/SapBERT-from-PubMedBERT-fulltext']
    maximum_candidates=200
    classifier_name=None

    umls = UMLS(mrconso_file)
    umls.augment_umls(new_auis_filename, 
                      output_dir, 
                      retriever_names,
                      maximum_candidates,
                      classifier_name)

if __name__ == "__main__":
    main()