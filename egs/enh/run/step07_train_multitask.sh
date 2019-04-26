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
dir=exp/train_7a_enh
tmp=/tmp/kaldi_feats.XXXXXX

. ./path.sh
. parse_options.sh || exit 1;

#Stage 1: feat preparation
featdir=feat_lps
if [ $stage -le 1 ]; then
    mkdir -p $dir || true
    cat $featdir/train_tr90/{noisy,clean}_lps.scp |\
    compute-cmvn-stats scp:- - |\
    cmvn-to-nnet --binary=false - $dir/tgt_lps.feature_transform || exit 1

    dim_lps=$(feat-to-dim --print-args=false scp:$featdir/train_tr90/clean_lps.scp -)
    utils/nnet/gen_splice.py --fea-dim=$dim_lps --splice=$splice --splice-step=1 > $dir/splice_lps.nnet
    nnet-concat --binary=false $dir/tgt_lps.feature_transform \
        $dir/splice_lps.nnet $dir/src.feature_transform || exit 1

    get-inv-tx --binary=false $dir/tgt_lps.feature_transform $dir/inv.feature_transform || exit 1
    
    dim_irm=$(feat-to-dim --print-args=false scp:$featdir/train_tr90/irm.scp -)
    python utils/nnet/gen_splice.py --fea-dim=$dim_irm --splice=0 > $dir/tgt_irm.feature_transform || exit 1

    dim_sum=$((dim_lps+dim_irm))
    echo "<ParallelComponent> <InputDim> $dim_sum <OutputDim> $dim_sum <NestedNnetFilename> \
        $dir/tgt_lps.feature_transform $dir/tgt_irm.feature_transform </NestedNnetFilename>" |\
    nnet-initialize --binary=false - $dir/tgt.feature_transform || exit 1

    ##create proto
    dim_feat=$(nnet-info $dir/src.feature_transform | grep output-dim | head -n1 | awk '{ print $NF }')
    python2 utils/nnet/make_nnet_proto.py --no-softmax $dim_feat $dim_sum $nn_depth $hid_dim > $dir/base.proto || exit 1
    python utils/nnet/gen_splice.py --fea-dim=$dim_lps --splice=0 > $dir/lps_out.nnet || exit 1  #dummy layer
    echo -e "<Nnet>\n<Sigmoid> $dim_irm $dim_irm\n</Nnet>" > $dir/irm_out.nnet || exit 1
    echo "<ParallelComponent> <InputDim> $dim_sum <OutputDim> $dim_sum <NestedNnetFilename> \
        $dir/lps_out.nnet $dir/irm_out.nnet </NestedNnetFilename>" |\
    nnet-initialize - - |\
    nnet-concat <(nnet-initialize $dir/base.proto - ) - $dir/nnet.init || exit 1
    
    ##insepction
    nnet-info $dir/nnet.init
fi

#Stage 2: dnn train
if [ $stage -le 2 ]; then
  feature_transform=$dir/src.feature_transform
  targets_transform=$dir/tgt.feature_transform
  nnet_init=$dir/nnet.init
  objective_function="multitask,mse,257,0.8,mse,257,0.2"
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)&
  $cuda_cmd $dir/log/train_nnet.log \
    local/train_enhancer.sh --feature-transform $feature_transform \
    --nnet-init $nnet_init --multitask true \
    --learn-rate $lr0 --copy-feats true --copy-feats-tmproot $tmp \
    --train-opts "--start-halving-impr 0.005 --end-halving-impr 0.0005 --max-iters $max_iters" \
    --train-tool "nnet-train-frmshuff-reg --objective-function=$objective_function" \
    --train-tool-opts "--minibatch-size=$bunch_size --momentum=0.9 --targets-transform=$targets_transform" \
    $featdir/train_tr90 $featdir/train_cv10 $dir || exit 1
    echo "Training done" && exit 0
fi

#Stage 3: dnn test
if [ $stage -le 3 ]; then
   local/eval_enhancement.sh --stage 0 --post-process true \
        --post-upper 0.9 --post-lower 0.6 $dir || exit 1 
   exit 0
fi

