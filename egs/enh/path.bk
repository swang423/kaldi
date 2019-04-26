case $(hostname) in 
    "cw1"|"cw01"|"cw2"|"cw02"|"cw3"|"cw03"|"cw4"|"cw04")
        export KALDI_ROOT=/home/swang423/kaldi18
        [ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
        export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$PWD:$PATH
        [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
        . $KALDI_ROOT/tools/config/common_path.sh
        ;;
    "pc2"|"pc02"|"pc1"|"pc01")
        echo "On $(hostname)"
        export KALDI_ROOT=/home/swang423/kaldi17/
        src=src
        export PATH=$PWD/utils/:$KALDI_ROOT/$src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/$src/fstbin/:$KALDI_ROOT/$src/gmmbin/:$KALDI_ROOT/$src/featbin/:$KALDI_ROOT/$src/lm/:$KALDI_ROOT/$src/sgmmbin/:$KALDI_ROOT/$src/sgmm2bin/:$KALDI_ROOT/$src/fgmmbin/:$KALDI_ROOT/$src/latbin/:$KALDI_ROOT/$src/nnetbin:$KALDI_ROOT/$src/nnet2bin/:$KALDI_ROOT/$src/kwsbin:$KALDI_ROOT/$src/smapbin/:$PWD:$PATH
        ;;
    *)
        echo "Kaldi not compiled" && exit 1
esac
export LC_ALL=C
