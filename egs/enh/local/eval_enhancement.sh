#!/bin/bash

## This script is used to evaluate the performance of DNN-based enhancement systems
## Stage 0: predict enhanced LPS features from noisy feature (feature extraction handled in earlier steps)
## Stage 1: LPS2WAV
## Stage 2: PESQ

## Author: Sicheng Wang @ GT (sichengwang.gatech@gmail.com)
## 4/11/2019

set -eu
stage=99
cmd=run.pl

post_process=false                          #If true, use irm postprocess 
post_lower=0.6                              #lower bound (epsilon)
post_upper=0.9                              #upper bound(gamma)
featdir=feat_lps                            #testing scp is at $featdir/test
srcwavdir=data_train100_repeat5/test/wav    #noisy wav for phs
srcscpdir=data_train100_repeat5/test        #noisy scp is at $srcscpdir/wav.scp
clnwavdir=data/test/wav                     #clean wav for reference
nnet=
lps2wavbin=bin/lps2wav
pesqbin=bin/pesq
lpsdim=257

echo "$0 $@"
[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;
if [ $# != 1 ]; then
    echo "usage: $0: <dir>"
    exit 1
fi
dir=$1
lpsdir=$(mktemp -d /tmp/eval_lps.XXXXXX)
outwavdir=$(mktemp -d /tmp/eval_wav.XXXXXX)
trap "echo '# Removing tmp files $lpsdir $outwavdir @ $(hostname)'; rm -r $lpsdir $outwavdir" EXIT
pesqlog=$dir/pesq.log
if [ -z $nnet ] ; then
    echo "No specified nnet. Using $dir/final.nnet"
    nnet=$dir/final.nnet
fi
for f in $nnet $dir/{src,inv}.feature_transform $featdir/test/noisy_lps.scp; do
    [ ! -f $f ] && echo "$0: missing $f" && exit 1
done
for d in $clnwavdir ; do
    [ ! -d $d ] && echo "$0: missing dir $d" && exit 1
done
echo "Dir: $dir"
echo "Nnet: $nnet"
# inference
if [ $stage -le 0 ] ; then
    mkdir -p $lpsdir || true
    feats="ark:copy-feats scp:$featdir/test/noisy_lps.scp ark:- |"
    if $post_process ; then
        arkdir=$(mktemp -d /tmp/eval_ark.XXXXXX)
        trap "echo '# Removing tmp files $arkdir @ $(hostname)'; rm -r $arkdir" EXIT
        nnet-forward --feature-transform=$dir/src.feature_transform --use-gpu=no \
            $nnet "$feats" ark:$arkdir/pred.ark || exit 1
        irm="ark:select-feats ${lpsdim}-$((lpsdim*2-1)) ark:$arkdir/pred.ark ark:- |"
        select-feats 0-$((lpsdim-1)) ark:$arkdir/pred.ark ark:- |\
        nnet-forward $dir/inv.feature_transform ark:- ark:- |\
        irm-post-process --lower-bound=$post_lower --upper-bound=$post_upper \
            "$feats" ark:- "$irm" ark:- |\
        copy-feats-to-htk --output-dir=$lpsdir --output-ext=lps \
            --sample-period=160000 ark:- || exit 1
    else
    nnet-forward --feature-transform=$dir/src.feature_transform --use-gpu=no \
        $nnet "$feats" ark:- |\
        select-feats 0-256 ark:- ark:- |\
        nnet-forward $dir/inv.feature_transform ark:- ark:- |\
        copy-feats-to-htk --output-dir=$lpsdir --output-ext=lps \
            --sample-period=160000 ark:- || exit 1
    fi
fi
# lps2wav
if [ $stage -le 1 ] ; then
    scp2wav=false
    if [ ! -d $srcwavdir ] ; then
        scp2wav=true
    else
        lpscount=`ls -l $lpsdir/*lps 2>/dev/null | wc -l`
        wavcount=`ls -1 $srcwavdir/*wav 2>/dev/null | wc -l`
        if [ $lpscount -ne $wavcount ] ; then
            scp2wav=true
        fi
    fi
    if $scp2wav ; then
        echo "Info: We are now going to write noisy speech in test to disk."
        if [ ! -f $srcscpdir/wav.scp ] ; then
            echo "Error: missing $srcscpdir/wav.scp required to write noisy speech." && exit 1
        else
            echo "Info:If the test set will be used over and over"
            echo "     Consider saving the source wavs to a permanent location"
            echo "     See option srcwavdir."
            srcwavdir="$(mktemp -d /tmp/eval_noisy_$(basename $srcscpdir).XXXXXX)"
            trap "echo '# Removing tmp wav $srcwavdir @ $(hostname)'; rm -r $srcwavdir" EXIT
            local/scp2sh.sh --stage 0 $srcscpdir/wav.scp $srcscpdir/wav.sh $srcwavdir || exit 1
            local/scp2sh.sh --stage 1 $srcscpdir/wav.scp $srcscpdir/wav.sh $srcwavdir || exit 1
        fi
    fi

    mkdir -p $outwavdir || true
    command -v $lps2wavbin >/dev/null 2>&1 || { echo >&2 "$lps2wavbin not found.  Aborting."; exit 1; }
    mkdir -p $outwavdir || true
    for f in $lpsdir/* ; do
        f=${f##*/}
        g=${f%.*}
        [ ! -f $srcwavdir/$g.wav ] && echo "Error: missing $srcwavdir/$g.wav" && exit 1
        $lps2wavbin -q $srcwavdir/$g.wav $lpsdir/$f $outwavdir/$g.wav || exit 1
    done
fi
# Eval PESQ
if [ $stage -le 2 ] ; then 
    command -v $pesqbin >/dev/null 2>&1 || { echo >&2 "$pesqbin not found.  Aborting."; exit 1; }
    for f in $outwavdir/* ; do
        g=${f##*/}
        g=${g%.*}
        g=${g%_*}
        g=${g%_*}
        ref=$clnwavdir/${g}.wav
        [ ! -f $ref ] && echo "$ref not found." && exit 1
        echo -n "$(basename $f ) " && $pesqbin +16000 $ref $f | tail -n1 
    done > $pesqlog
    ncount=$(grep -w Prediction $pesqlog | wc -l)
    if [ $ncount -le 0 ] ; then
        echo "Did not get any count?: $ncount" && exit 1
    fi
    nsum=$(grep -w Prediction $pesqlog | awk -F ' ' '{ print $NF }' |\
        paste -sd+ | bc)
    mean_pesq=$(echo "$nsum/$ncount" | bc -l)
    echo "Mean pesq: $mean_pesq of $ncount files." && exit 0
fi
