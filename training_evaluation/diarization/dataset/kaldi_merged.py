def load_reco2dur(reco2dur_file):
    reco2dur = {}
    with open(reco2dur_file, 'r') as f:
        for line in f:
            utt, dur = line.strip().split()
            reco2dur[utt] = float(dur)
    return reco2dur

def create_segments(merge_file, reco2dur_file, output_file):
    reco2dur = load_reco2dur(reco2dur_file)
    
    with open(merge_file, 'r') as merge_f, open(output_file, 'w') as out_f:
        for line in merge_f:
            merged_id, *utts = line.strip().split()
            start_time = 0.0

            for utt in utts:
                duration = reco2dur.get(utt, 0.0)
                end_time = start_time + duration
                
                out_f.write(f"{utt} {merged_id} {start_time:.2f} {end_time:.2f}\n")
                
                start_time = end_time


if __name__ == "__main__":
    # modify which_dataset = (train,validation,test)
    which_dataset = "train"

    merge_file = f'merged_audio/{which_dataset}/details/merge'
    reco2dur_file = f'va_filtered_audio/details/reco2dur'
    output_file = f'merged_audio/{which_dataset}/details/segments'
    create_segments(merge_file, reco2dur_file, output_file)
