python3 main.py --do_predict --output_dir out/scorer --model classifier --predict_file data/hotpot-all/dev.json,comparison,bridge,intersec,onehop --init_checkpoint model/scorer/best-model.pt --max_seq_length 400 --prefix dev_
echo "Checkpoint 1"
python3 show_result.py --data_file dataset/hotpot_dev_distractor_v1.json --prediction_file prediction_$1.json