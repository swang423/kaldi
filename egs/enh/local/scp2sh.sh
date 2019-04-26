#!/bin/bash

# Convert a scp with PIPED wav (e.g. sph2wav, etc..) to bash script to generate the wav

stage=0
lazy=true
split=false
. ./path.sh
. parse_options.sh || exit 1;
if [ $# -gt 1 -a $# -le 3 ] ; then
    scp=$1
    sh=$2
    dir=${3:-tmp}
elif [ $# -eq 1 ]; then
    base=$1
    scp=$base/wav.scp
    sh=$base/wav.sh
    dir=$base/wav
else
	echo "$0 --lazy false <scp-in> <sh-out> [<wav-dir>]"
    echo "$0 --lazy true <scp-dir>"
	exit 1
fi

if [ $stage -le 0 ]; then
paste -d ' ' \
	<(cut -d ' ' -f 2- $scp | sed 's,- |$,,g' | sed 's,|$,,g' ) \
	<(cut -d ' ' -f 1 $scp | sed "s,^,$dir/,g" | sed 's,$,.wav,g') \
	> $sh || exit 1
    if $split ; then
        split -d -a 1 -l$((`wc -l <$sh`/8)) $sh $sh.
        fc=$(ls $sh.? | wc -l)
        echo "Split into $fc parts. $sh.? are ready to run. Remember to run stage 1"
        for f in $sh.? ; do
            chmod 744 $f
        done
        mv $sh.0 $sh.9
    else
        chmod 744 $sh
        echo "$sh is ready to run. Remember to run stage 1 afterwards." 
    fi
[ -d $dir ] || mkdir -p $dir
exit 0
fi

if [ $stage -le 1 ] ; then
    if $split ; then
        [ -z $base ] && echo "Error: base needs to be set" && exit 1
        mkdir -p $base/log
        utils/run.pl JOB=1:9 $base/log/pipe2wav.JOB.log \
            sh $base/wav.sh.JOB || exit 1
    else
        sh $sh || exit 1
    fi
    echo " Remember to run stage 2 afterwards."
    exit 0
fi
# Get new wav.scp
if [ $stage -le 2 ] ; then
	num=$(ls $dir/ | wc -l)
	if [ $num -le 0 ] ; then
		echo "Error: Did you run $sh to generate the wav?"
		exit 1
	fi
#	scp_dir=${scp%/*}
	if [ ! -e $scp ] ; then
		echo "Moving old scp to $scp.old"
		cp $scp $scp.old
	fi
	paste -d ' ' \
		<(ls $dir/ | awk -F '/' '{ print $NF }' | sed 's,.wav$,,g') \
		<(ls $dir/ | sed "s,^,$dir/,g") |\
	sort > $scp
    echo "Writing new scp to $scp"
fi
