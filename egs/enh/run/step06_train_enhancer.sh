#!/bin/bash
. ./cmd.sh
. ./path.sh

stage=9
nn_depth=3
hid_dim=2048
splice=5 
bunch_size=32
lr0=1e-5
max_iters=50
dir=exp/train_6a_enh
tmp=/tmp/kaldi_feats.XXXXXX     #you may need to change this to a tmp dir on your system

. ./path.sh
. parse_options.sh || exit 1;

#Stage 1: feat preparation
featdir=feat_lps
if [ $stage -le 1 ]; then
    mkdir -p $dir || true
    cat $featdir/train_tr90/{noisy,clean}_lps.scp |\
    compute-cmvn-stats scp:- - |\
    cmvn-to-nnet --binary=false - $dir/tgt.feature_transform || exit 1

    dim_lps=$(feat-to-dim --print-args=false scp:$featdir/train_tr90/clean_lps.scp -)
    utils/nnet/gen_splice.py --fea-dim=$dim_lps --splice=$splice --splice-step=1 > $dir/splice_lps.nnet
    nnet-concat --binary=false $dir/tgt.feature_transform \
        $dir/splice_lps.nnet $dir/src.feature_transform || exit 1

    get-inv-tx --binary=false $dir/tgt.feature_transform $dir/inv.feature_transform || exit 1
#    exit 0
fi

#Stage 2: dnn train
if [ $stage -le 2 ]; then
  feature_transform=$dir/src.feature_transform
  targets_transform=$dir/tgt.feature_transform
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)&
  $cuda_cmd $dir/log/train_nnet.log \
    local/train_enhancer.sh --feature-transform $feature_transform \
    --hid-layers $nn_depth --hid-dim $hid_dim --splice $splice --proto-opts "--no-softmax" \
    --learn-rate $lr0 --copy-feats true --copy-feats-tmproot $tmp \
    --train-opts "--start-halving-impr 0.005 --end-halving-impr 0.0005 --max-iters $max_iters" \
    --train-tool "nnet-train-frmshuff-reg --objective-function=mse" \
    --train-tool-opts "--minibatch-size=$bunch_size --momentum=0.9 --targets-transform=$targets_transform" \
    $featdir/train_tr90 $featdir/train_cv10 $dir || exit 1
    echo "Training done" && exit 0
fi

#Stage 3: dnn test
if [ $stage -le 3 ]; then
   local/eval_enhancement.sh --stage 0 $dir || exit 1 
   exit 0
fi

