#!/bin/bash

## This version uses overlap-add to replace lps2wav
## Hence, we only require src scp instead of src wav on disk

set -eu
stage=99
splice=0

clnwavdir=data/test/wav #clean directory
srcscp=data_noisex92_repeat5/test/wav.scp       #scp of noisy wav
featdir=feat_lps                                #noisy input (feature that go into nnet)
testdir=noisex92                                #[test | test_match | aurora4 | noisex92 ] must match $srcwavdir and $clnwavdir
srcwavtype=noisy                                #[noisy | clean ] noisy for enhancement, clean for synthesis

cmd=run.pl
mode=lps                                        #[lps | fft] if fft, then must give phs feature in htk format
pesq=/home/swang423/cdr/PESQ/P862/Software/source/pesq

nnet=               #must be given
feature_transform=  #if none, try to use $dir/src.feature_transform
inverse_transform=  #if none, try to use $dir/inv.feature_transform
pesqlog=            #default to $dir/pesq.log
lsdlog=             #If nonempty, measure utt level lsd
stoilog=            #If nonempty, measure utt level stoi
subsamp=0           #subsampling for debugging
python_opts=        #see nnet-forward-mse.py
outwavdir=          #if specified, then run pesq using the wav in the dir (useful for resuming from step 2)
keeptmp=false       #if true, then keep predicted lps and synthesized wav
skipcheck=false     #if true, skip checking for required for (useful for just running later stages)
skippesq=false      #if true, skip stage 2 and go straight to stoi eval

## This section contains some "advanced" use for the script 
## Basically variations of experiment that use more features or require extra post-processing steps
sigmoid_postpp=false    #if true, use sigmoid-mask-post-process,see lps-sigmoid-mask-post-processing
multifeature=false      #for clean+noise+snr feat in for syntheis
irmout=false            #if true, the predicted output is interpreted as irm

hostname
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
    echo "usage: $0: <dir>"
    echo "  --stage                                  # starts from which stage"
    echo "  --lsdlog <lsdlog>                        #if provided, measure utt level lsd"
    echo "  --nnet <nnet.h5>                         #if provided, use nnet"
    exit 1
fi
dir=$1

echo "Evaluating on $(hostname)"
echo "#Dir: $dir"
echo "#Nnet : $nnet"

###########################
# check for required files
##########################
if ! $skipcheck ; then
if [ -z $outwavdir ] ; then
    outwavdir="$(mktemp -d /tmp/eval_wav.XXXXXX)" && echo "Creating tmp wav dir $outwavdir"
    trap "echo '# Removing generated wav in $outwavdir @ $(hostname)'; rm -r $outwavdir" EXIT
fi
[ -z $nnet ] && nnet=$dir/final.nnet.h5
[ -z $pesqlog ] && pesqlog=$dir/pesq.log
[ -z $feature_transform ] && feature_transform=$dir/src.feature_transform
[ -z $inverse_transform ] && inverse_transform=$dir/inv.feature_transform

for f in $nnet $feature_transform $featdir/$testdir/${srcwavtype}_lps.scp ; do
    [ ! -f $f ] && echo "$0: missing $f" && exit 1
done
if $multifeature ; then
#    for f in $featdir/$testdir/{noise_lps.scp,snr.ali} $dir/{noise,snr}.feature_transform ; do
    for f in $featdir/$testdir/bnf.scp $dir/noise.feature_transform ; do
        [ ! -f $f ] && echo "$0: Multifeature selected. missing $f" && exit 1
    done
fi
if [ ! -f $inverse_transform ] ; then
    [ ! -f $dir/tgt.feature_transform ] && echo "$0: missing $dir/tgt.feature_transform to produce inverse transform." && exit 1
    get-inv-tx $dir/tgt.feature_transform $inverse_transform || exit 1
fi
for bin in $pesq ; do
    command -v $bin >/dev/null 2>&1 || { echo >&2 "$bin not found.  Aborting."; exit 1; }
done

fi

###########################
# noisy wav to clean lps
##########################
if [ $stage -le 0 ] ; then
#mkdir -p $lpsdir || true
feats="ark:copy-feats scp:$featdir/$testdir/${srcwavtype}_lps.scp ark:- |"
if [ $subsamp -gt 0 ] ; then
    feats="$feats subset-feats --n=$subsamp ark:- ark:- |"
fi
if [ $splice -gt 0 ] ; then
    feats="$feats splice-feats --left-context=$splice --right-context=$splice ark:- ark:- |"
fi
feats="$feats nnet-forward $feature_transform ark:- ark:- |"
if $multifeature ; then
#    feats_noise="ark:copy-feats scp:$featdir/$testdir/noise_lps.scp ark:- | nnet-forward $dir/noise.feature_transform ark:- ark:- |"
#    feats_snr="ark:ali-to-post ark:$featdir/$testdir/snr.ali ark:- | post-to-feats --post-dim=5 ark:- ark:- | nnet-forward $dir/snr.feature_transform ark:- ark:- |"
#    feats="$feats paste-feats-sql --length-tolerance=0 ark:- \"$feats_noise\" \"$feats_snr\" ark:- |"
    feats_noise="ark:copy-feats scp:$featdir/$testdir/bnf.scp ark:- | nnet-forward $dir/noise.feature_transform ark:- ark:- |"
    feats_snr="ark:ali-to-post ark:$featdir/$testdir/snr.ali ark:- | post-to-feats --post-dim=5 ark:- ark:- | nnet-forward $dir/snr.feature_transform ark:- ark:- |"
    feats="$feats paste-feats-sql --length-tolerance=0 ark:- \"$feats_noise\" \"$feats_snr\" ark:- |"
#    feats="$feats paste-feats-sql --length-tolerance=0 ark:- \"$feats_noise\" ark:- |"
fi
#feats="ark: copy-feats ark:$my.ark ark:- | nnet-forward $dir/inv.feature_transform ark:- ark:- |"
feats="$feats python steps_kt/nnet-forward-mse.py $python_opts $nnet |"
feats="$feats select-feats 0-256 ark:- ark:- |"
if $sigmoid_postpp ; then
    feats="$feats lps-sigmoid-mask-post-processing ark:- scp:$featdir/$testdir/${srcwavtype}_lps.scp ark:- |"
elif $irmout ; then
    feats="$feats apply-mask --ignore-range=true scp:$featdir/$testdir/${srcwavtype}_lps.scp ark:- ark:- |"
else
    feats="$feats nnet-forward $inverse_transform ark:- ark:- |"
fi
    if [ ! -z $lsdlog ] ; then
        [ ! -f $featdir/$testdir/clean_lps.scp ] && echo "Missing $featdir/$testdir/clean_lps.scp" && exit 1
        eval-loss-lsd scp:$featdir/$testdir/clean_lps.scp "$feats" ark,t:$lsdlog || exit 1
        ncount=$(cat $lsdlog | wc -l)
        if [ $ncount -le 0 ] ; then
            echo "Did not get any count?: $ncount" && exit 1
        fi
        nsum=$(cat $lsdlog | awk -F ' ' '{ print $NF }' |\
            paste -sd+ | bc)
        mean_lsd=$(echo "$nsum/$ncount" | bc -l)
        echo "Mean LSD: $mean_lsd " && exit 0
    else
        compute-phs-feats --dither=0 --frame-length=32 --frame-shift=16 \
            --window-type="hamming" --remove-dc-offset=false --preemphasis-coefficient=0 \
            scp:$srcscp ark:- |\
        overlap-add --window-shift=256 --fft-size=512 --window-type="hamming" \
            --output-dir=$outwavdir --fs=16000 "$feats" ark:- || exit 1
    fi
fi
###########################
#lps2wav
##########################
if [ $stage -eq 1 ]; then
    echo "Info: Stage 1 is merged into stage 0."
    echo "Info: We will proceed to stage 2." 
fi
###########################
#pesq
##########################
if [ $stage -le 2 ]; then
    if [ ! -z $pesqlog ] ; then
    looperr=false
    for f in $outwavdir/* ; do
        g=${f##*/}
        if [ $srcwavtype == "noisy" ] ; then
            g=${g%.*}
            if [ $testdir == "aurora4" ] ; then
                g=${g%?} #trim last char
            else #timit
                g=${g%_*}
                g=${g%_*}
            fi
            ref=$clnwavdir/${g}.wav
        elif [ $srcwavtype == "clean" ] ; then
            ref=$srcwavdir/$g
        else
            echo "Invalid srcwavtype: $srcwavtype " && exit 1
        fi
        [ ! -f $ref ] && looperr=true && break
        echo -n "$(basename $f) " && $pesq +16000 $ref $f | tail -n1
    done > $pesqlog
    if $looperr ; then
        echo "$ref not found." && exit 1
    fi
    ncount=$(grep -w Prediction $pesqlog | wc -l)
    if [ $ncount -le 0 ] ; then
        echo "Did not get any count?: $ncount" && exit 1
    fi
    nsum=$(grep -w Prediction $pesqlog | awk -F ' ' '{ print $NF }' |\
        paste -sd+ | bc)
    mean_pesq=$(echo "$nsum/$ncount" | bc -l)
    if $keeptmp ; then
        echo "Info: tmpwav on $(hostname) at $outwavdir"
    else
        if [ -d $outwavdir ] ; then
            rm -r $outwavdir
        fi  
    fi  
#    echo "Mean pesq: $mean_pesq" && exit 0
    echo "Mean pesq: $mean_pesq"
    fi
    if [ -z $stoilog ] ; then
        exit 0
    fi
fi
###########################
#STOI
##########################
if [ $stage -le 3 ]; then
    MATLAB_PATH=/usr/local/MATLAB/R2018b/bin
    #require pip install resampy
    #also pystoi has lower values cf. matlab stoi
    #see steps_kt/eval_stoi.py for usage
    if [ ! -z $stoilog ] ; then
    $MATLAB_PATH/matlab -r "addpath('steps_kt');eval_stoi('$clnwavdir','$outwavdir','$stoilog'); quit"
    #matlab -r "addpath('steps_kt');try eval_stoi('$clnwavdir','$outwavdir','$stoilog'); catch; end; quit"
    echo "" && exit 0
    fi
    exit 0
fi
###########################
#results by SNR
##########################
if [ $stage -le 4 ] ; then
    for f in $lsdlog $pesqlog $stoilog ; do
    if [ -z $f ] && [ ! -f $f ]; then
        echo "Missing $lsdlog, $pesqlog, or $stoilog" && exit 1
    fi
    done
    echo -e "\t<LSD>\t<PESQ>\t<STOI>"
    for snr in `seq -6 2 16`; do #test snr
        lsdsum=$(grep "SNR$snr" $lsdlog | awk -F ' ' '{ print $NF }' |\
            paste -sd+ | bc)
        pesqsum=$(grep -w Prediction $pesqlog | grep "SNR$snr" |\
            awk -F ' ' '{ print $NF }' | paste -sd+ | bc)
        stoisum=$(grep "SNR$snr" $stoilog | awk -F ' ' '{ print $NF }' |\
            paste -sd+ | bc)
        snrcount=$(grep "SNR$snr" $lsdlog | wc -l)
        lsdmean=$(echo "scale=2; $lsdsum/$snrcount" | bc)
        pesqmean=$(echo "scale=2; $pesqsum/$snrcount" | bc)
        stoimean=$(echo "scale=2; $stoisum/$snrcount" | bc)
        echo -e "SNR$snr\t$lsdmean\t$pesqmean\t$stoimean"
    done
    exit 0
fi

###########################
#original noisy vs clean
##########################
if [ $stage -le -1 ]; then
    pesqlog=tmp/pesq.log
    for f in $srcwavdir/* ; do
        g=${f##*/}
        g=${g%.*}
        g=${g%_*}
        g=${g%_*}
        ref=$clnwavdir/${g}.wav
        [ ! -f $ref ] && echo "$ref not found." && exit 1
        echo -n "$(basename $f) " && $pesq +16000 $ref $f | tail -n1
    done > $pesqlog
    ncount=$(grep -w Prediction $pesqlog | wc -l)
    if [ $ncount -le 0 ] ; then
        echo "Did not get any count?: $ncount" && exit 1
    fi
    nsum=$(grep -w Prediction $pesqlog | awk -F ' ' '{ print $NF }' |\
        paste -sd+ | bc)
    mean_pesq=$(echo "$nsum/$ncount" | bc -l)
    echo "Mean pesq: $mean_pesq" && exit 0
    #2.32826829268292682926    
fi
echo "Done" && exit 0
