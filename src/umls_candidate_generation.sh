for date in 2021AA;
do

    python umls_augmentation.py '../data/umls_versions/'$date'-ACTIVE/META_DL/MRCONSO_MASTER.RRF' '../data/insertion_sets/'$date'_insertion.txt' ../output

done