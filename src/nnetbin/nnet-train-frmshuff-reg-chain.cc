// nnetbin/nnet-train-frmshuff.cc
// loop training

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
namespace kaldi {
/*
   This function converts comma-spearted string into float vector.
*/
void ReadCommaSeparatedCommand(const std::string &s,
                                std::vector<BaseFloat> *v) {
  std::vector<std::string> split_string;
  SplitStringToVector(s, ",", true, &split_string);
  for (size_t i = 0; i < split_string.size(); i++) {
    float ret;
    ConvertStringToReal(split_string[i], &ret);
    v->push_back(ret);
  }
}
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  nnet-train-frmshuff [options] <paired-src-rspecifier> <paired-tgt-rspecifier> <unpaired-src-rspecifier> <unpaired-tgt-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-frmshuff scp:feature.scp ark:posterior.ark scp:noisy.scp scp:clean.scp nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
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
    std::string targets_transform;
    po.Register("targets-transform", &targets_transform, "Targets transform in Nnet format");
    std::string objective_function_f = "multitask";
    po.Register("objective-function-forward", &objective_function_f, "Objective function : xent|mse");
    std::string objective_function_b = "multitask";
    po.Register("objective-function-backward", &objective_function_b, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    std::string loss_weight;
    po.Register("loss-weight",&loss_weight,"a comma separated string of 4 real values of Noisy2Clean(NC), CN, NN, and CC loss respectively.");
    po.Read(argc, argv);

    if (po.NumArgs() != 8-(crossvalidate?2:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      unpaired_feature_rspecifier = po.GetArg(3),
      unpaired_targets_rspecifier = po.GetArg(4),
      model_filename0 = po.GetArg(5),
      model_filename1 = po.GetArg(6);
    //model_filename0 uses src feature transform
    //model_filename1 uses tgt feature transform        
    std::string target_model_filename0, target_model_filename1;
    if (!crossvalidate) {
      target_model_filename0 = po.GetArg(7);
      target_model_filename1 = po.GetArg(8);
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
    Nnet nnet_transt;
    if (targets_transform != "") {
      nnet_transt.Read(targets_transform);
    }

    Nnet nnet0,nnet1;
    nnet0.Read(model_filename0);
    nnet1.Read(model_filename1);
    nnet0.SetTrainOptions(trn_opts);
    nnet1.SetTrainOptions(trn_opts);//if need different optim, use a 2nd trn_opts

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet0.SetDropoutRate(0.0);
      nnet1.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader0(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader0(targets_rspecifier);
    SequentialBaseFloatMatrixReader feature_reader1(unpaired_feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader1(unpaired_targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      KALDI_ERR << "Not implemented"; //we are going to use frame weights to differentiate paired vs unpaired utterances
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer0(rnd_opts);
    MatrixRandomizer targets_randomizer0(rnd_opts);
    MatrixRandomizer feature_randomizer1(rnd_opts);
    MatrixRandomizer targets_randomizer1(rnd_opts);
    VectorRandomizer weights_randomizer0f(rnd_opts);
    VectorRandomizer weights_randomizer0b(rnd_opts);
    VectorRandomizer weights_randomizer1f(rnd_opts);
    VectorRandomizer weights_randomizer1b(rnd_opts);

    MultiTaskLoss multitask0f(loss_opts),multitask0b(loss_opts);
    MultiTaskLoss multitask1(loss_opts),multitask2(loss_opts);
    if (0 == objective_function_f.compare(0, 9, "multitask")) {
      multitask0f.InitFromString(objective_function_f);
      multitask2.InitFromString(objective_function_f);
    }
    if (0 == objective_function_b.compare(0, 9, "multitask")) {
      multitask0b.InitFromString(objective_function_b);
      multitask1.InitFromString(objective_function_b);
    }
    std::vector<BaseFloat> loss_weight_vector;
    if (!loss_weight.empty()) {
      ReadCommaSeparatedCommand(loss_weight, &loss_weight_vector);
    }else{
      for (int32 kk = 0 ; kk < 4 ; kk ++ )
        loss_weight_vector.push_back(1.0);
    }    
    KALDI_ASSERT(loss_weight_vector.size()==4); //we defined 4 losses: Noisy2Clean(NC), CN, CC, NN
    CuMatrix<BaseFloat> feats_transf, nnet_out_f,nnet_out_b, tgt_transf, obj_diff_f,obj_diff_b;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    int32 num_done_feat = 0 , num_done_tgt = 0;
    bool shuff0 = false, shuff1f = false, shuff1b = false;
    while ( !(feature_reader0.Done() || targets_reader0.Done()) || !feature_reader1.Done() || !targets_reader1.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      for ( ; !feature_reader0.Done() && !targets_reader0.Done(); 
            feature_reader0.Next(), targets_reader0.Next()) {
        if (feature_randomizer0.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader0.Key();
        if (targets_reader0.Key() != utt) {
            KALDI_WARN << "utt mismatch for " << utt;
            num_no_tgt_mat++;
            continue;
        }
        Matrix<BaseFloat> mat = feature_reader0.Value();
        Matrix<BaseFloat> targets = targets_reader0.Value();
        Vector<BaseFloat> weights;
        {
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(targets.NumRows());
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(targets.NumRows() != min) targets.Resize(min, targets.NumCols(), kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << targets.NumRows()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }
        weights.Resize(mat.NumRows());
        weights.Set(loss_weight_vector[0]);
        //By now all mat and targets are matched (with zero fillers)
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
        nnet_transt.Feedforward(CuMatrix<BaseFloat>(targets), &tgt_transf);

        KALDI_ASSERT(feats_transf.NumRows() == tgt_transf.NumRows());
        feature_randomizer0.AddData(feats_transf);
        targets_randomizer0.AddData(tgt_transf);
        weights_randomizer0f.AddData(weights);
        weights.Set(loss_weight_vector[1]);
        weights_randomizer0b.AddData(weights);
        shuff0 = true;
        num_done++;
      }
      for (;!feature_reader1.Done(); feature_reader1.Next()){
        if (feature_randomizer1.IsFull()) break; 
        Matrix<BaseFloat> mat = feature_reader1.Value();
        Vector<BaseFloat> weights(mat.NumRows());
        weights.Set(loss_weight_vector[2]);
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
        feature_randomizer1.AddData(feats_transf);
        weights_randomizer1f.AddData(weights);
        shuff1f = true;
        num_done_feat++;
      }
      for (;!targets_reader1.Done(); targets_reader1.Next()){
        if (targets_randomizer1.IsFull()) break; 
        Matrix<BaseFloat> targets = targets_reader1.Value();
        Vector<BaseFloat> weights(targets.NumRows());
        weights.Set(loss_weight_vector[3]);// This could potentially serve as loss weights
        nnet_transt.Feedforward(CuMatrix<BaseFloat>(targets), &tgt_transf);
        targets_randomizer1.AddData(tgt_transf);
        weights_randomizer1b.AddData(weights);
        shuff1b = true;
        num_done_tgt ++;
      }
      // randomize
      if (!crossvalidate && randomize) {
        if (shuff0){
        const std::vector<int32>& mask0 = randomizer_mask.Generate(feature_randomizer0.NumFrames());
        feature_randomizer0.Randomize(mask0);
        targets_randomizer0.Randomize(mask0);
        weights_randomizer0f.Randomize(mask0);
        weights_randomizer0b.Randomize(mask0);
        shuff0 = false;
        }
        if(shuff1f){
        const std::vector<int32>& mask1 = randomizer_mask.Generate(feature_randomizer1.NumFrames());
        feature_randomizer1.Randomize(mask1);
        weights_randomizer1f.Randomize(mask1);
        shuff1f = false;
        }
        if(shuff1b){
        const std::vector<int32>& mask2 = randomizer_mask.Generate(targets_randomizer1.NumFrames());
        targets_randomizer1.Randomize(mask2);
        weights_randomizer1b.Randomize(mask2);
        shuff1b = false;
        }
      }
      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer0.Done() || !feature_randomizer1.Done() || !targets_randomizer1.Done();) {
        /////////////////////////////////////////////////////////////////////
        //parallel
        /////////////////////////////////////////////////////////////////////
        if(!feature_randomizer0.Done()){
            const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer0.Value();
            const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer0.Value();
            const Vector<BaseFloat>& frm_weights_f = weights_randomizer0f.Value();
            const Vector<BaseFloat>& frm_weights_b = weights_randomizer0b.Value();

            nnet0.Propagate(nnet_in, &nnet_out_f);
            nnet1.Propagate(nnet_tgt, &nnet_out_b);
            multitask0f.Eval(frm_weights_f, nnet_out_f, nnet_tgt, &obj_diff_f); 
            multitask0b.Eval(frm_weights_b, nnet_out_b, nnet_in, &obj_diff_b); 
            if (!crossvalidate) {
              nnet0.Backpropagate(obj_diff_f, NULL);
              nnet1.Backpropagate(obj_diff_b, NULL);
            }

            total_frames += nnet_in.NumRows();
            feature_randomizer0.Next();
            targets_randomizer0.Next();
            weights_randomizer0f.Next();
            weights_randomizer0b.Next();
        }
        /////////////////////////////////////////////////////////////////////
        //feat only
        /////////////////////////////////////////////////////////////////////
        if(!feature_randomizer1.Done()){
            const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer1.Value();
            const Vector<BaseFloat>& frm_weights = weights_randomizer1f.Value();

            nnet0.Propagate(nnet_in, &nnet_out_f);
            nnet1.Propagate(nnet_out_f, &nnet_out_b);
            multitask1.Eval(frm_weights, nnet_out_b, nnet_in, &obj_diff_b); 
            if (!crossvalidate) {
              nnet1.Backpropagate(obj_diff_b, NULL);
            }

            total_frames += nnet_in.NumRows();
            feature_randomizer1.Next();
            weights_randomizer1f.Next();
        }
        /////////////////////////////////////////////////////////////////////
        //tgt only
        /////////////////////////////////////////////////////////////////////
        if(!targets_randomizer1.Done()){
            const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer1.Value();
            const Vector<BaseFloat>& frm_weights = weights_randomizer1b.Value();

            nnet1.Propagate(nnet_tgt, &nnet_out_b);
            nnet0.Propagate(nnet_out_b, &nnet_out_f);
            multitask2.Eval(frm_weights, nnet_out_f, nnet_tgt, &obj_diff_f); 
            if (!crossvalidate) {
              nnet0.Backpropagate(obj_diff_f, NULL);
            }

            total_frames += nnet_tgt.NumRows();
            targets_randomizer1.Next();
            weights_randomizer1b.Next();
        }
      } //batch training loop
    }//main loop
    
    if (!crossvalidate) {
      nnet0.Write(target_model_filename0, binary);
      nnet1.Write(target_model_filename1, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

  KALDI_LOG << multitask0f.Report();
  KALDI_LOG << multitask0b.Report();
  KALDI_LOG << multitask1.Report();
  KALDI_LOG << multitask2.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
