#!/bin/bash

## This script aligns clean targets with noisy features
## Contact: Sicheng Wang (sichengwang.gatech@gmail.com)
## 4/11/2019
set -eu

######### CONFIG ##############
stage=9
datanoisydir=data_train100_repeat5  #output of step01
featnoisydir=feat_train100_repeat5  #output of step02
featcleandir=feat_clean             #output of step03
featdir=feat_lps                    #where to write the final features
feattype=lps
include_clean=false                 #If true, include clean in training too
######## END OF CONFIG ###########
. path.sh
. parse_options.sh

if [ $stage -le 0 ] ; then
    for x in train_{tr90,cv10} test ; do
    mkdir -p $datanoisydir/$x/tmp
    cut -d '_' -f 1,2 $datanoisydir/$x/wav.scp > $datanoisydir/$x/tmp/clean.list || exit 1
    done
fi

if [ $stage -le 1 ] ; then
    for x in train_{cv10,tr90} test ; do
        mkdir -p $featdir/$x || true
        #clean
        f=$datanoisydir/$x/tmp/clean.list
        g=$featcleandir/$x/$feattype.scp
        [ ! -f $f ] && echo "Error:Require $f from stage 0." && exit 1
        [ ! -f $g ] && echo "Error:Require $g from step01." && exit 1
        while read l ; do
            grep -w $l $g
        done < $f > $featdir/$x/clean_$feattype.scp || exit 1
    done
fi

#change utt names
if [ $stage -le 2 ] ; then
    for x in train_{tr90,cv10} test ; do
        for y in clean ; do
          cp $featdir/$x/${y}_$feattype.scp $featdir/$x/${y}_$feattype.scp.bk
        paste <(cut -d ' ' -f 1 $datanoisydir/$x/wav.scp) \
              <(cut -d ' ' -f 2 $featdir/$x/${y}_$feattype.scp.bk) \
              > $featdir/$x/${y}_$feattype.scp.tmp || exit 1
        done
#        #noisy
        if $include_clean && [ "$x" != "test" ]; then
            cat $featnoisydir/$x/$feattype.scp $featcleandir/$x/$feattype.scp \
                > $featdir/$x/noisy_$feattype.scp || exit 1
            cat $featdir/$x/clean_$feattype.scp.tmp $featcleandir/$x/$feattype.scp \
                > $featdir/$x/clean_$feattype.scp || exit 1
        else
            mv $featdir/$x/clean_$feattype.scp.tmp $featdir/$x/clean_$feattype.scp || exit 1
            cp $featnoisydir/$x/$feattype.scp $featdir/$x/noisy_$feattype.scp || exit 1
        fi
        for f in bk tmp ; do
        [ -f $featdir/$x/${y}_$feattype.scp.$f ] && rm $featdir/$x/${y}_$feattype.scp.$f
        done
    done
fi

