#!/bin/bash

## This script uses a customized kaldi tool to synthesize noisy speech with one additive
## noise at specified SNR intervals. 
## User needs to provide clean speech (specified as wav.scp in kaldi format)
## and noise waveforms (a directory containing the wav files)
## It should be run from an environment that resembles a typical kaldi exp 
## like timit/s5 that contains utils and steps

## Author: Sicheng Wang (sichengwang.gatech@gmail.com)
## 4/11/2019

set -eu

######### CONFIG ##############
mode=train                          #[train|test]
data_clean=data                     #where clean speech is
                                    #it should have wav.scp under $data_clean/{train,test}
snrs=(-5 0 5 10 15 20)              #SNR levels in dB
repeat=5                            #This is an "approximate" multiplicative factor
                                    #i.e. for 1 hr of clean speech in $data_clean, it will
                                    #output $repeat hrs of synthetic noisy speech
data_noise=data/noise               #directory containing all the additive noise wav
data=data_train100_repeat5          #output directory
add_noise_tool=
pipe=false                          #if wav.scp is specified in a piped format
######## END OF CONFIG ###########

. path.sh
. parse_options.sh

if [ $# -gt 1 ] ; then
    echo "Usage: $0:"
    echo "  --mode <train|test>     generating training or test data set"
    echo "The output will default to $data/train or $data/test"
    exit 1
fi

[ -z $add_noise_tool ] && add_noise_tool=$KALDI_ROOT/src/featbin/wav-reverberate-equal
                                    #This tool works by adding noise to speech by trimming
                                    #off remaining long noise and repeating short noise
                                    #so ideally noise in $data_noise should have about
                                    #the same length
command -v $add_noise_tool >/dev/null 2>&1 || { echo >&2 "$add_noise_tool not found.  Aborting."; exit 1; }
mkdir -p $data/noise || true
mkdir -p $data/$mode/tmp || true
noise_list=$data/$mode/tmp/noise.list
snr_list=$data/noise/snr.list
ls $data_noise/*.wav | sed 's/.wav$//g' | sed "s,$data_noise/,,g" > $noise_list || exit 1
echo ${snrs[@]} | tr ' ' '\n' > $snr_list || exit 1

for f in $data_clean/$mode/{wav.scp,utt2spk} $noise_list $snr_list ; do
    [ ! -f $f ] && echo "Error: missing $f." && exit 1
done
num_col=$(awk '{print NF}' $data_clean/$mode/wav.scp | sort -nu | tail -n1)
if [ $num_col -lt 2 ] ; then
    echo "$0: Expect wav.scp to have at least 2 columns." && exit 1
elif [ $num_col -gt 2 ]; then
    if ! $pipe ; then
        echo "$0: wav.scp has more than 2 columns"
        echo "If wav.scp specified as a piped process as typical for kaldi"
        echo "Use flag \"--pipe true\"" && exit 1
    fi
fi

mkdir -p $data/$mode/{list,tmp} || true
cut -d ' ' -f 1 $data_clean/$mode/wav.scp > $data/$mode/tmp/clean.list.head || exit 1
cut -d ' ' -f 2- $data_clean/$mode/wav.scp > $data/$mode/tmp/clean.list.tail || exit 1
num_utt=$(cat $data/$mode/tmp/clean.list.head | wc -l)
num_noise_tr=$(cat $noise_list | wc -l)
num_snr=${#snrs[@]}
echo "Info: $num_noise_tr noise, $num_snr snr levels chosen."
repeat_noise=$(echo "$num_utt/$num_noise_tr+1" | bc)
repeat_snr=$(echo "$num_utt/$num_snr+1" | bc)
if [ $repeat_noise -lt 1 ] || [ $repeat_snr -lt 1 ] ; then
echo "Error: Math error: $repeat_noise,$repeat_snr" && exit 1
fi
for kk in `seq 0 $((repeat-1))` ; do
    cat `yes $noise_list | head -n$repeat_noise | paste -s` |\
        shuffle_list.pl --srand $kk | head -n $num_utt > $data/$mode/tmp/noise.$kk || exit 1
    cat `yes $snr_list | head -n$repeat_snr | paste -s` |\
        shuffle_list.pl --srand $((kk*2)) | head -n $num_utt > $data/$mode/tmp/snr.$kk || exit 1
    paste -d '_' $data/$mode/tmp/clean.list.head $data/$mode/tmp/noise.$kk <(sed 's/^/SNR/g' $data/$mode/tmp/snr.$kk) > $data/$mode/tmp/scp.$kk.head || exit 1
    if $pipe ; then
    paste -d ' ' $data/$mode/tmp/clean.list.tail \
        <(sed "s,^,$add_noise_tool --print-args=false --additive-signals=\"$data_noise/,g" $data/$mode/tmp/noise.$kk | sed "s,$,.wav\",g")\
        <(sed "s,^,--snrs=\",g" $data/$mode/tmp/snr.$kk) |\
        sed "s,$,\" --start-times=\"0\" - - |,g" > $data/$mode/tmp/scp.$kk.tail || exit 1
    else
    paste -d ' ' \
        <(sed "s,^,$add_noise_tool --print-args=false --additive-signals=\"$data_noise/,g" $data/$mode/tmp/noise.$kk | sed "s,$,.wav\",g") \
        <(sed "s,^,--snrs=\",g" $data/$mode/tmp/snr.$kk | sed "s,$,\" --start-times=\"0\",g") \
        <(sed "s,$, - |,g" $data/$mode/tmp/clean.list.tail) > $data/$mode/tmp/scp.$kk.tail || exit 1
    fi
paste -d ' ' $data/$mode/tmp/scp.$kk.{head,tail} > $data/$mode/tmp/scp.${kk}.part
done
cat $data/$mode/tmp/scp.*.part > $data/$mode/tmp/wav.scp

## This part is optional, but utt2spk will be handy later on
true && \
{
for y in utt2spk text stm; do
if [ -e $data_clean/$mode/$y ] ; then
    if [ "$y" == "stm" ] ; then
        tail -n +4 $data_clean/$mode/stm | cut -d ' ' -f 2- > $data/$mode/list/${y}.part || exit 1
    else
        cut -d ' ' -f 2- $data_clean/$mode/$y > $data/$mode/list/${y}.part || exit 1
    fi
  cut -d ' ' -f 1 $data/$mode/tmp/wav.scp |\
   paste - <(cat `yes $data/$mode/list/${y}.part | head -n$repeat | paste -s`) |\
   sort -k 1 | uniq > $data/$mode/$y || exit 1
    if [ "$y" == "stm" ] ; then
        head -n3 $data_clean/$mode/stm | cat - $data/$mode/stm > $data/$mode/stm.tmp
        mv $data/$mode/stm.tmp $data/$mode/stm
    fi
fi
done
    
if [ -e $data/$mode/utt2spk ] ; then
utt2spk_to_spk2utt.pl $data/$mode/utt2spk > $data/$mode/spk2utt || exit 1
fi
for f in spk2gender glm ; do
    if [ -e $data_clean/$mode/$f ] ; then
    cp $data_clean/$mode/$f $data/$mode/$f
    fi
done
}

sort -k 1 $data/$mode/tmp/wav.scp | uniq > $data/$mode/wav.scp || exit 1
echo "Done." && exit 0
