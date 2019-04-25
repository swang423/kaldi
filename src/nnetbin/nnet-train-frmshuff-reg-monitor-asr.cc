// nnetbin/nnet-train-frmshuff.cc
// read in an asr and monitor xent loss

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
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

    std::string corpus="wsj";
    po.Register("corpus", &corpus, "wsj|timit");
    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");
//    std::string tgt_weighting;
//    po.Register("tgt-weight", &tgt_weighting, "Weight for different dimensions of the output.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    int32 asr_layer = 12;
    po.Register("asr-layer", &asr_layer, "where to split nnet and branch into asr");
 
    po.Read(argc, argv);

    if (po.NumArgs() != 6-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
        ali_rspecifier = po.GetArg(3),
      model_filename = po.GetArg(4),
        asr_model_filename = po.GetArg(5);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(6);
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

    Nnet nnet,asr;
    nnet.Read(model_filename);
    asr.Read(asr_model_filename);
    KALDI_ASSERT(nnet.NumComponents() >= asr_layer);
    Nnet nnet_bot(nnet);
    Nnet nnet_top(nnet);
    for (int32 kk = 0 ; kk < asr_layer; kk++){
        nnet_top.RemoveComponent(0);
    }
    for (int32 kk = asr_layer; kk < nnet.NumComponents(); kk++){
        nnet_bot.RemoveLastComponent();
    }
    nnet_bot.SetTrainOptions(trn_opts);
    nnet_top.SetTrainOptions(trn_opts);

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    RandomAccessPosteriorReader ali_reader(ali_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    PosteriorRandomizer post_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent(loss_opts);
    Mse mse(loss_opts);
    MultiTaskLoss multitask(loss_opts); 
    CuMatrix<BaseFloat> feats_transf, nnet_bot_out, nnet_top_out,tgt_transf, obj_diff;
    CuMatrix<BaseFloat> asr_out,nnet_bot_diff, asr_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0, num_no_post = 0;
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
		std::size_t pos_underscore = utt.find("_"); //for wsj
        if(corpus == "timit")
    		pos_underscore = utt.find("_",pos_underscore+1); //for timit
		KALDI_ASSERT(pos_underscore!=std::string::npos);
		std::string utt_clean = utt.substr(0,pos_underscore);
        if(!ali_reader.HasKey(utt_clean)){
          KALDI_WARN << utt << ", missing alignment";
          num_no_post++;
          continue;
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
        Posterior post = ali_reader.Value(utt_clean);
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
          lenght.push_back(post.size());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(targets.NumRows() != min) targets.Resize(min, targets.NumCols(), kCopyData);
            if(post.size() != min) post.resize(min);
            if(weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << targets.NumRows()
                        << ", post " << post.size()
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
        post_randomizer.AddData(post);
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
        post_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          post_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer.Value();
        const Posterior& posterior = post_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass
        nnet_bot.Propagate(nnet_in, &nnet_bot_out);
        nnet_top.Propagate(nnet_bot_out,&nnet_top_out);
        asr.Propagate(nnet_bot_out,&asr_out);

/*        KALDI_LOG << posterior[0][0].first << posterior[1][0].first
                << posterior[100][0].first << posterior[101][0].first;
        KALDI_LOG << asr_out.Range(0,2,0,3);
        KALDI_LOG << asr_out.Range(100,2,0,3);
        return 0;
*/
        //
        xent.Eval(frm_weights, asr_out, posterior, &asr_diff); 
        mse.Eval(frm_weights, nnet_top_out, nnet_tgt, &obj_diff);
        // backward pass
        if (!crossvalidate) {
          // backpropagate
          nnet_top.Backpropagate(obj_diff, &nnet_bot_diff);
          nnet_bot.Backpropagate(nnet_bot_diff, NULL);
        }
        
        total_frames += nnet_in.NumRows();
      }
    }
    
    if (!crossvalidate) {
      nnet_bot.AppendNnet(nnet_top);
      nnet_bot.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with no post, " << num_no_post
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    KALDI_LOG << xent.Report();
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
