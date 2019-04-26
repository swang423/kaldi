Speech enhancement based on spectral mapping

This is a bundled software to train a DNN-based speech enhancement 
network under the Kaldi framework.
Detailed description can be found in each steps under run/*.sh

1)  Download and compile Kaldi @ https://github.com/kaldi-asr/kaldi
2)  Compile the tools under src by merging them into Kaldi.
    This could be accomplished by moving the source codes into
    kaldi's src, update the Make files in each directory featbin 
    and nnetbin, then make. The codes are developed in Kaldi release
    back in 2017 and 2018.
3)  This directory should sit parallel with $KALDI_ROOT/egs/wsj/s5. 
    It should have access to utils/, steps/, cmd.sh, and path.sh.
    You may need to update the search path in path.sh
    To test your environment is set up correctly, try
    . ./path.sh
    nnet-info
    If you see "Print human-readable information ...", then the 
    search path is set correctly.
4)  step01-step04 prepares the multi-condition training data
    It will output single channel speech corrupted by one additive 
    noise a user provides at specified SNRs at random. The extracted
    features will be saved at feat_lps by default. Detailed instruct-
    ions can be found in each step.
5)  step05 extracts optional IRM feature.
6)  step06 trains and evaluates a basic DNN.
7)  step07 trains DNN with IRM as the secondary task. 
    (https://arxiv.org/pdf/1703.07172.pdf)
    It also support IRM post process. 
8)  There is a small amount of sample data under data/
    You may get a feel of the whole process first by running through
    all the scripts in this order:
    [a]
    bash run/step01_synthesis.sh --mode train
    bash run/step01_synthesis.sh --mode test
    By now there will be a new directory data_train100_repeat5/ contain-
    ing all the noisy wav scp (in scripted format rather than actual
    files on disk)
    [b]
    bash run/step02_extract_noisy_feats.sh --stage 0
    By now there will be a new directory feat_train100_repeat5/ contain-
    ing all the noisy LPS features in kaldi ark.
    In later experiments, you may want to set flag --cv-portion=10 or
    --cv-portion 5.
    [c]
    bash run/step03_extract_clean_feats.sh --stage 0
    By now there will be a new directory feat_clean/ containing all the
    clean LPS features in kaldi ark.
    --cv-portion should match that in 8[b]
    [d]
    bash run/step04_align_clean.sh --stage 0
    This will create a new directory feat_lps/ containing the paired
    noisy and clean feature scp files. 
    [e]
    bash run/step05_extract_irm_feats.sh --stage 0
    This will extract IRM features and add them to feat_lps/
    This is optional and is only used in run/step07_train_multitask.sh
    [f]
    bash run/step06_train_enhancer.sh --stage 1 --max-iters 2
    This will try to train a baseline DNN system. We train only 2 epochs
    to make sure the scripts run first. You may need to modify the flag
    --tmp to a tmp directory on your system.
    If it runs successfully, there should be a final.nnet under
    exp/train_6a_enh/
    bash run/step06_train_enhancer.sh --stage 3 
    This evaluates the trained DNN with PESQ as the selected metric.
    [g]
    bash run/step07_train_multitask.sh --stage 1 --max-iters 2
    This trains a multitask DNN with IRM outputs too.
    The weights of LPS and IRM targets could be adjusted.
    It should render a system with lower MSE.
    bash run/step07_train_multitask.sh --stage 3 
    This will evaluate the trained multitask DNN with IRM post-process
    The thresholds could be experimentally tuned. See --post-upper and 
    --post-lower in local/eval_enhancement.sh
9)  Train a desirable DNN using your audio data
    What audio files you'll need and how to prepare them for the project?
    - Sample audios are provided under data/
    - data/train and data/test have the same structure. Each contains
        wav.scp and utt2spk. 
        wav.scp contains the information on the clean utterance. 
        It follows Kaldi's convention in the format of
        <audio-name>    <location on disk | pipe>
        uttspk is used to split the wav.scp for parallelism. It is not
        strictly required. You may create a dummy uttspk by ordering
        the audio sequentially
        <audio-name>    <audio-index>
    - data/noise contains noise waveforms to be mixed with the clean
        audio. They should be about the same length as the clean audio.
    - To create a mismatched testing set, the user may wish to put the
        mismatched noise in another directory like data/noise_unseen
        and call run/step01_synthesis with flag:
        --data-noise data/noise_unseen
    - Repeat steps in run/ to train a DNN-based enhancement system.
       
Updated 4/18/2019 
