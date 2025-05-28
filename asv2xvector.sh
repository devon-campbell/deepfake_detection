#!/bin/bash

# Navigate to ASVspoof2021_DF_eval directory
cd export/corpora/asv2021/ASVspoof2021_DF_eval/flac

# Create wav.scp from flac files
for file in *.flac; do
    utt_id=$(basename $file .flac)
    echo "$utt_id flac -c -d -s $file |" >> ../../../../../asvspoof_xvector/eval/asvspoof2021/wav.scp
done

# Navigate to ASVspoof xvector directory
cd ../../../../../asvspoof_xvector/eval/asvspoof2021

# Generate utt2spk and spk2utt files
awk '{print $1, $1}' wav.scp > utt2spk
../../utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt

# Validate speech files
cd ../../..
utils/validate_data_dir.sh --no-feats --no-text asvspoof_xvector/eval/asvspoof2021

# Generate MFCCs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 20 --cmd "run.pl" asvspoof_xvector/eval/asvspoof2021

# Compute VADs
sid/compute_vad_decision.sh --nj 20 --cmd "run.pl" asvspoof_xvector/eval/asvspoof2021

# Extract xvectors
sid/nnet3/xvector/extract_xvectors.sh \
  --cmd "run.pl --mem 4G" \
  exp/xvector_nnet_1a \
  asvspoof_xvector/eval/asvspoof2021 \
  asvspoof_xvector/eval/asvspoof2021/xvectors

# Navigate to ASVspoof2019_LA_train directory
cd export/corpora/asv2019/LA/ASVspoof2019_LA_train/flac

# Create wav.scp from flac files
for file in *.flac; do
    utt_id=$(basename "$file" .flac)
    echo "$utt_id sox $(pwd)/$file -t wav -r 16000 -b 16 - |" >> ../../../../../asvspoof_xvector/train/asvspoof2019/wav.scp
done

# Navigate to ASVspoof2019 directory
cd ../../../../../asvspoof_xvector/train/asvspoof2019

# Generate utt2spk and spk2utt files
awk '{print $1, $1}' wav.scp > utt2spk
../../utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt

# Validate speech files
cd ../../..
utils/validate_data_dir.sh --no-feats --no-text asvspoof_xvector/train/asvspoof2019

# Generate MFCCs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 20 --cmd "run.pl" asvspoof_xvector/train/asvspoof2019

# Compute VADs
sid/compute_vad_decision.sh --nj 20 --cmd "run.pl" asvspoof_xvector/train/asvspoof2019

# Extract xvectors
sid/nnet3/xvector/extract_xvectors.sh \
  --cmd "run.pl --mem 4G" \
  exp/xvector_nnet_1a \
  asvspoof_xvector/train/asvspoof2019 \
  asvspoof_xvector/train/asvspoof2019/xvectors

# Navigate to ASVspoof2019_LA_dev directory
cd export/corpora/asv2019/LA/ASVspoof2019_LA_dev/flac

# Create wav.scp from flac files
for file in *.flac; do
    utt_id=$(basename "$file" .flac)
    echo "$utt_id sox $(pwd)/$file -t wav -r 16000 -b 16 - |" >> ../../../../../asvspoof_xvector/dev/asvspoof2019/wav.scp
done

# Navigate to ASVspoof2019 directory
cd ../../../../../asvspoof_xvector/dev/asvspoof2019

# Generate utt2spk and spk2utt files
awk '{print $1, $1}' wav.scp > utt2spk
../../utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt

# Validate speech files
cd ../../..
utils/validate_data_dir.sh --no-feats --no-text asvspoof_xvector/dev/asvspoof2019

# Generate MFCCs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 20 --cmd "run.pl" asvspoof_xvector/dev/asvspoof2019

# Compute VADs
sid/compute_vad_decision.sh --nj 20 --cmd "run.pl" asvspoof_xvector/dev/asvspoof2019

# Extract xvectors
sid/nnet3/xvector/extract_xvectors.sh \
  --cmd "run.pl --mem 4G" \
  exp/xvector_nnet_1a \
  asvspoof_xvector/dev/asvspoof2019 \
  asvspoof_xvector/dev/asvspoof2019/xvectors

# Navigate to ASVspoof2017_V2_train directory
cd export/corpora/asv2017/ASVspoof2017_V2_train

# Create wav.scp from wav files
for file in *.wav; do
    utt_id=$(basename "$file" .wav)
    echo "$utt_id sox $(pwd)/$file -t wav -r 16000 -b 16 - |" >> ../../../../../asvspoof_xvector/train/asvspoof2017/wav.scp
done

# Navigate to ASVspoof2017 directory
cd ../../../../../asvspoof_xvector/train/asvspoof2017

# Generate utt2spk and spk2utt files
awk '{print $1, $1}' wav.scp > utt2spk
../../utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt

# Validate speech files
cd ../../..
utils/validate_data_dir.sh --no-feats --no-text asvspoof_xvector/train/asvspoof2017

# Generate MFCCs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 20 --cmd "run.pl" asvspoof_xvector/train/asvspoof2017

# Compute VADs
sid/compute_vad_decision.sh --nj 20 --cmd "run.pl" asvspoof_xvector/train/asvspoof2017

# Extract xvectors
sid/nnet3/xvector/extract_xvectors.sh \
  --cmd "run.pl --mem 4G" \
  exp/xvector_nnet_1a \
  asvspoof_xvector/train/asvspoof2017 \
  asvspoof_xvector/train/asvspoof2017/xvectors

#!/bin/bash

# Navigate to ASVspoof2017_V2_dev directory
cd export/corpora/asv2017/ASVspoof2017_V2_dev

# Create wav.scp from wav files
for file in *.wav; do
    utt_id=$(basename "$file" .wav)
    echo "$utt_id sox $(pwd)/$file -t wav -r 16000 -b 16 - |" >> ../../../../../asvspoof_xvector/dev/asvspoof2017/wav.scp
done

# Navigate to ASVspoof2017 directory
cd ../../../../../asvspoof_xvector/dev/asvspoof2017

# Generate utt2spk and spk2utt files
awk '{print $1, $1}' wav.scp > utt2spk
../../utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt

# Validate speech files
cd ../../..
utils/validate_data_dir.sh --no-feats --no-text asvspoof_xvector/dev/asvspoof2017

# Generate MFCCs
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 20 --cmd "run.pl" asvspoof_xvector/dev/asvspoof2017

# Compute VADs
sid/compute_vad_decision.sh --nj 20 --cmd "run.pl" asvspoof_xvector/dev/asvspoof2017

# Extract xvectors
sid/nnet3/xvector/extract_xvectors.sh \
  --cmd "run.pl --mem 4G" \
  exp/xvector_nnet_1a \
  asvspoof_xvector/dev/asvspoof2017 \
  asvspoof_xvector/dev/asvspoof2017/xvectors