#!/bin/sh

## This script computes spectral features for the synthesized speech
## Contact: Sicheng Wang (sichengwang.gatech@gmail.com)
## 4/11/2019

set -eu

######### CONFIG ##############
stage=9
nj=2                            #run parallel with $nj jobs
cv_portion=50                   #split train into training and validation as a percentage
                                #normally we use 10 percent
                                #in practice this should be about 10 or 5
datadir=data_train100_repeat5   #output from step01 script
featdir=feat_train100_repeat5   #where to write feature

feature_type=lps
frame_length=32
frame_shift=16
######## END OF CONFIG ###########
. path.sh
. cmd.sh
. parse_options.sh

if [ $# -gt 1 ] ; then
    echo "Usage: $0:"
    echo "  --stage     resume from stage x"
    exit 1
fi

#split train into 90/10
if [ $stage -le 0 ] ; then
    if [ $cv_portion -gt 20 ] ; then
        echo "Info: The cv_portion is set usually large. Consider lower it to 10 or 5 for actual data."
    fi
    train=$datadir/train
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent $cv_portion $train ${train}_tr90 ${train}_cv10 || exit 1
fi

## Extract noisy
if [ $stage -le 1 ] ; then
    echo " ==== Running extraction of $feature_type feats with: ===="
    echo " Window length: $frame_length ms."
    echo " Window shift: $frame_shift ms."
    if [ $nj -le 10 ] ; then
        decimal=1
    elif [ $nj -le 100 ] ; then
        decimal=2
    else
        echo "$0: Requested more than 100 jobs. Check?" && exit 1
    fi
    for x in train_cv10 train_tr90 test ; do
        if [ ! -f $datadir/$x/wav.scp ] ; then
            echo "$0: missing $datadir/$x/wav.scp" && exit 1
        fi
        mkdir -p $datadir/$x/split$nj || true
        split -a $decimal --additional-suffix=.scp --numeric-suffixes=1 -n l/$nj \
           $datadir/$x/wav.scp $datadir/$x/split$nj/wav. || exit 1
        for t in log data; do
            mkdir -p $featdir/$x/${feature_type}_$t || true
        done
        $train_cmd JOB=1:$nj $featdir/$x/${feature_type}_log/extract_noisy_${feature_type}.JOB.log \
        compute-lps-feats \
            --dither=0 --frame-length=$frame_length --frame-shift=$frame_shift \
            --window-type="hamming" --remove-dc-offset=false --preemphasis-coefficient=0 \
            scp:$datadir/$x/split$nj/wav.JOB.scp ark:- \| \
        copy-feats --compress=true ark:- \
            ark,scp:$PWD/$featdir/$x/${feature_type}_data/feats.JOB.ark,$featdir/$x/${feature_type}_data/feats.JOB.scp || exit 1
        cat $featdir/$x/${feature_type}_data/feats*.scp > $featdir/$x/${feature_type}.scp
    done
fi
exit 0
