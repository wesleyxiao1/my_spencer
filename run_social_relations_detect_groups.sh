python3 social_relations.py --input_file=data/group_track_data.csv --output_file=data/social_relations_data3.csv --num_frames_for_speed=1
python3 detect_groups.py --input_file=data/group_track_data.csv --social_relations=data/social_relations_data3.csv --output_file=data/detected_groups3.csv