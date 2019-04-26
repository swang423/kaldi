#!/bin/sh

## This script computes spectral features for the synthesized speech
## Contact: Sicheng Wang (sichengwang.gatech@gmail.com)
## 4/11/2019

set -eu
. ./path.sh
. ./cmd.sh

######### CONFIG ##############
stage=9
cv_portion=50                   #split train into training and validation as a percentage
                                #in practice this should be about 10 or 5
datadir=data                    #where clean data is
featdir=feat_clean              #where we'll write the feature

feature_type=lps
frame_length=32
frame_shift=16
######## END OF CONFIG ###########
. path.sh
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

## Extract clean
if [ $stage -le 1 ]; then
    for x in train_{tr90,cv10} test ; do
        if [ ! -f $datadir/$x/wav.scp ] ; then
            echo "$0: missing $datadir/$x/wav.scp" && exit 1
        fi
        mkdir -p $featdir/$x || true
        compute-lps-feats \
            --dither=0 --frame-length=$frame_length --frame-shift=$frame_shift \
            --window-type="hamming" --remove-dc-offset=false --preemphasis-coefficient=0 \
            scp:$datadir/$x/wav.scp ark:- |\
        copy-feats --compress=true ark:- \
            ark,scp:$PWD/$featdir/$x/${feature_type}.ark,$featdir/$x/${feature_type}.scp || exit 1
    done
fi


