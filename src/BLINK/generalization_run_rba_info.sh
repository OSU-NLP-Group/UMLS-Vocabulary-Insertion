model_name=$1

echo 'Testing Model '$model_name
for version in 2021AA 2021AB 2022AA 2022AB;
do
    data='models/uva_ins_'$version'_max_len_64_num_cands_50_train_size_all_sorted_cuis_rba_info/top50_candidates/'
    output='models/uva_ins_'$version'_max_len_64_num_cands_50_train_size_all_sorted_cuis_rba_info'

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python blink/crossencoder/train_cross.py --path_to_model $model_name --data_path $data --output_path $output --learning_rate 2e-05 --num_train_epochs 0 --max_seq_length 128 --max_context_length 64 --max_cand_length 64 --train_batch_size 1 --eval_batch_size 128 --top_k 50 --bert_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --eval_interval 5000 --type_optimization all_encoder_layers --add_linear --data_parallel --debug
done