// nnetbin/nnet-train-frmshuff.cc

// use asr as secondary task
//enh use lps
//asr use log fbank; must be single splice input
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-linear-transform.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-frmshuff scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts,freeze_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string targets_transform,inverse_transform;
    po.Register("targets-transform", &targets_transform, "Targets transform in Nnet format");
    po.Register("inverse-transform", &inverse_transform, "Inverse targets transform in Nnet format");
    std::string fbank_transform;
    po.Register("fbank-transform", &fbank_transform, "fbank transform in Nnet format");
    std::string fbank_conf;
    po.Register("fbank-conf",&fbank_conf,"mel filter bank");
    std::string objective_function = "mse";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string asr_nnet = "";
    po.Register("asr-nnet", &asr_nnet, "load asr as secondary loss");
//    std::string asr_write = "";
//    po.Register("asr-write", &asr_write, "write asr for debugging");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
   
    BaseFloat weight = 0;
    po.Register("weight", &weight, "weight of asr task. Large if using post softmax (xent), small for other hidden layers (mse)"); 
    bool debug = false;
    po.Register("debug", &debug, "if true, no MSE task");
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
    //targets_rspecifier should be pasted feats: lps followed by mfcc most likely    
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    Nnet nnet_transt, inv_transf;
    if (targets_transform != "") {
      nnet_transt.Read(targets_transform);
    }
    if (inverse_transform != ""){
      inv_transf.Read(inverse_transform);
    }
    Nnet fbank_transf;
    if (fbank_transform != "") {
      fbank_transf.Read(fbank_transform);
    }


    KALDI_ASSERT(asr_nnet != "");

    Nnet nnet,asr,linker;
    nnet.Read(model_filename);
    asr.Read(asr_nnet);
/*
    KALDI_ASSERT(fbank_conf != "");
    Matrix<BaseFloat> mel_fbank;
    ReadKaldiObject(fbank_conf,&mel_fbank);
    KALDI_ASSERET(nnet.OutputDim() == mel_fbank.NumCols());
    KALDI_ASSERET(asr.InputDim() == mel_fbank.NumRows());

    {
        linker.AppendNnet(inv_transf);
        ExpComponent exp_layer(nnet.OutputDim(),nnet.OutputDim());
        linker.AppendComponent(exp_layer);
        LinearTransform fb_layer(nnet.OutputDim(),nnet.OutputDim());
        fb_layer.SetLinearity(mel_fbank);
        linker.AppendComponent(fb_layer);
        LogComponent log_layer(asr.InputDim(),asr.InputDim());
        linker.AppendComponent(log_layer);    
        linker.AppendNnet(asr);
        asr = linker;
    }
*/
    freeze_opts.learn_rate = 0;
    nnet.SetTrainOptions(trn_opts);
    asr.SetTrainOptions(freeze_opts);

    int32 enh_out_dim = nnet.OutputDim(),
          asr_in_dim = asr.InputDim();
    KALDI_ASSERT(asr_in_dim < enh_out_dim);
    int32 enh_feats_dim = enh_out_dim - asr_in_dim;

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      asr.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent(loss_opts);
    Mse mse(loss_opts),mse_asr(loss_opts);

    CuMatrix<BaseFloat> feats_transf, nnet_out, tgt_transf, enh_diff,in_diff;
    CuMatrix<BaseFloat> nnet_out_enh,nnet_out_asr,nnet_tgt_enh, nnet_feats_asr;
    CuMatrix<BaseFloat> asr_tgt, asr_out, asr_diff;
    nnet_out_enh.Resize(rnd_opts.minibatch_size,enh_feats_dim);
    nnet_out_asr.Resize(rnd_opts.minibatch_size,asr_in_dim);
    nnet_tgt_enh.Resize(rnd_opts.minibatch_size,enh_feats_dim);
    nnet_feats_asr.Resize(rnd_opts.minibatch_size,asr_in_dim);

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (!feature_reader.Done() && !targets_reader.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !feature_reader.Done() && !targets_reader.Done(); feature_reader.Next(), targets_reader.Next()) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets
        if (targets_reader.Key() != utt) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          break;
        }
        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Matrix<BaseFloat> targets = targets_reader.Value();
        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // correct small length mismatch ... or drop sentence
        {
          // add lengths to vector
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(targets.NumRows());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(targets.NumRows() != min) targets.Resize(min, targets.NumCols(), kCopyData);
            if(weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << targets.NumRows()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }
        // apply optional feature transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
        nnet_transt.Feedforward(CuMatrix<BaseFloat>(targets), &tgt_transf);

        // pass data to randomizers
        KALDI_ASSERT(feats_transf.NumRows() == tgt_transf.NumRows());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(tgt_transf);
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      // randomize
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        nnet.Propagate(nnet_in, &nnet_out);
//        nnet_out_enh.CopyFromMat(nnet_out.ColRange(0,enh_feats_dim));
        nnet_out_asr.CopyFromMat(nnet_out.ColRange(enh_feats_dim,asr_in_dim));
//        nnet_tgt_enh.CopyFromMat(nnet_tgt.ColRange(0,enh_feats_dim));
        nnet_feats_asr.CopyFromMat(nnet_tgt.ColRange(enh_feats_dim,asr_in_dim));

        if(crossvalidate)
          mse.Eval(frm_weights, nnet_out.ColRange(0,enh_feats_dim), nnet_tgt.ColRange(0,enh_feats_dim), &enh_diff);
        else
          mse.Eval(frm_weights, nnet_out, nnet_tgt, &enh_diff);

        asr.Feedforward(nnet_feats_asr,&asr_tgt);
        asr.Propagate(nnet_out_asr,&asr_out);
        if (objective_function == "xent") {
          xent.Eval(frm_weights, asr_out, asr_tgt, &asr_diff); 
        } else if (objective_function == "mse") {
          mse_asr.Eval(frm_weights, asr_out, asr_tgt, &asr_diff);
        } else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }
        if (!crossvalidate) {
          asr.Backpropagate(asr_diff, &in_diff);
          if(debug){
            enh_diff.SetZero();
          }
          enh_diff.ColRange(enh_feats_dim,asr_in_dim).AddMat(weight,in_diff);
          nnet.Backpropagate(enh_diff, NULL);
        }

        
        total_frames += nnet_in.NumRows();
/*
        if(total_frames > 320){
            return 0;   
        }else{
            in_diff.ApplyPowAbs(1.0,false);
            BaseFloat in_diff_sum = in_diff.Sum();
            enh_diff.ColRange(enh_feats_dim,asr_in_dim).AddMat(-weight,in_diff);
            enh_diff.ColRange(enh_feats_dim,asr_in_dim).ApplyPowAbs(1.0,false);
            BaseFloat enh_diff_sum = enh_diff.ColRange(enh_feats_dim,asr_in_dim).Sum();
            KALDI_LOG << "Frame: " << total_frames << "; enh/asr: " << enh_diff_sum << ", " << in_diff_sum;
        }
*/
      }
    }
    

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
//      if(asr_write != "")
//        asr.Write(asr_write,binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  
    if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse_asr.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }
    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
