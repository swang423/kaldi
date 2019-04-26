#!/bin/sh

## This script computes spectral features for the added noise in the synthesized speech
## Contact: Sicheng Wang (sichengwang.gatech@gmail.com)
## 4/11/2019

set -eu

######### CONFIG ##############
stage=9
nj=2                            #run parallel with $nj jobs
cv_portion=50                   #split train into training and validation as a percentage
                                #in practice this should be about 10 or 5
datadir=data_train100_repeat5   #output from step01 script
featnoisydir=feat_train100_repeat5   #where to write the noise feature, we set it to be same as noisy
featdir=feat_lps                #where to write the final features

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
    train=$datadir/train
    if [ ! -d ${train}_tr90 ] ; then
        utils/subset_data_dir_tr_cv.sh --cv-spk-percent $cv_portion $train ${train}_tr90 ${train}_cv10 || exit 1
    fi
fi

## Extract noise
if [ $nj -le 10 ] ; then
    decimal=1
elif [ $nj -le 100 ] ; then
    decimal=2
else
    echo "$0: Requested more than 100 jobs. Check?" && exit 1
fi
if [ $stage -le 1 ] ; then
    echo " ==== Running extraction of $feature_type feats with: ===="
    echo " Window length: $frame_length ms."
    echo " Window shift: $frame_shift ms."
    for x in train_cv10 train_tr90 test ; do
        if [ ! -f $datadir/$x/wav.scp ] ; then
            echo "$0: missing $datadir/$x/wav.scp" && exit 1
        fi
        mkdir -p $datadir/$x/split$nj || true
        sed 's/wav-reverberate-equal/wav-reverberate-diff/g' $datadir/$x/wav.scp \
            > $datadir/$x/noise.scp || exit 1
        split -a $decimal --additional-suffix=.scp --numeric-suffixes=1 -n l/$nj \
           $datadir/$x/noise.scp $datadir/$x/split$nj/noise. || exit 1
        for t in log data; do
            mkdir -p $featnoisydir/$x/${feature_type}_$t || true
        done
        $train_cmd JOB=1:$nj $featnoisydir/$x/${feature_type}_log/extract_noise_${feature_type}.JOB.log \
        compute-lps-feats \
            --dither=0 --frame-length=$frame_length --frame-shift=$frame_shift \
            --window-type="hamming" --remove-dc-offset=false --preemphasis-coefficient=0 \
            scp:$datadir/$x/split$nj/noise.JOB.scp ark:- \| \
        copy-feats --compress=true ark:- \
            ark,scp:$PWD/$featnoisydir/$x/${feature_type}_data/noise.JOB.ark,$featnoisydir/$x/${feature_type}_data/noise.JOB.scp || exit 1
        cat $featnoisydir/$x/${feature_type}_data/noise.*.scp > $featnoisydir/$x/noise_${feature_type}.scp
    done
fi

## Extract IRM
if [ $stage -le 2 ] ; then
    for x in train_cv10 train_tr90 test ; do
        for t in log data; do
            mkdir -p $featnoisydir/$x/irm_$t || true
        done
        for f in $featdir/$x/clean_${feature_type}.scp $featnoisydir/$x/noise_${feature_type}.scp ; do
            if [ ! -f $f ] ; then
                echo "$0: Missing $f" && exit 1
            fi
        done
        $train_cmd $featnoisydir/$x/irm_log/extract_irm.log \
        compute-irm-feats scp:$featdir/$x/clean_${feature_type}.scp \
            scp:$featnoisydir/$x/noise_${feature_type}.scp ark:- \| \
        copy-feats --compress=true ark:- \
            ark,scp:$PWD/$featnoisydir/$x/irm_data/irm.ark,$featdir/$x/irm.scp || exit 1
    done
fi
exit 0
