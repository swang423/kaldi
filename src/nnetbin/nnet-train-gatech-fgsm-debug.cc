// nnetbin/nnet-train-frmshuff.cc

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

#include <iostream>
#include <fstream>
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
    std::string objective_function = "mse";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights,
        "Per-utterance weights, used to re-scale frame-weights.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    
    float grad_step = 0.01,error_step = 1;
    po.Register("grad-step", &grad_step, "step size in FGSM");
    po.Register("error-step", &error_step, "step size multiplier for xent task");
    po.Read(argc, argv);
    

    std::string Errroratio = po.GetArg(1);
    float error_ratio=atof(Errroratio.c_str());

    std::string feature_rspecifier = po.GetArg(2),
      targets_rspecifier = po.GetArg(3),
      cluster_targets_rspecifier = po.GetArg(4),
      model_filename = po.GetArg(5),
      regression_filename = po.GetArg(6),
      cluster_filename = po.GetArg(7);

    std::string target_model_filename, target_regression_model_filename, target_cluster_model_filename;
     if (!crossvalidate) {
      target_model_filename = po.GetArg(8);
      target_regression_model_filename = po.GetArg(9);
      target_cluster_model_filename = po.GetArg(10);
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

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);
   
    Nnet regression_nnet;
    regression_nnet.Read(regression_filename);
    regression_nnet.SetTrainOptions(trn_opts);

    Nnet cluster_nnet;
    cluster_nnet.Read(cluster_filename);
    cluster_nnet.SetTrainOptions(trn_opts);


    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
	  regression_nnet.SetDropoutRate(0.0);
      cluster_nnet.SetDropoutRate(0.0);

    }

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    RandomAccessPosteriorReader cluster_targets_reader(cluster_targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatVectorReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }


    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    PosteriorRandomizer cluster_targets_randomizer(rnd_opts);
	VectorRandomizer utt_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Xent xent(loss_opts);
    Xent xent_adv(loss_opts);
    Mse mse(loss_opts);
    Mse mse_adv(loss_opts); 
    CuMatrix<BaseFloat> feats_transf, tgt_transf, nnet_out,
                        regression_nnet_out, cluster_nnet_out,nnet_in_copy,
                        obj_diff, regression_obj_diff, cluster_obj_diff,
                        regression_in_diff, cluster_in_diff, nnet_in_diff,
                        regression_in_diff_copy, rand_in_diff;


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
          continue;
        }
	    
        if (!cluster_targets_reader.HasKey(utt)) {
                                        KALDI_WARN << utt << ", missing cluster targets";
                                        num_no_tgt_mat++;
                                        continue;
        }
        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
	     if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        }

        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Matrix<BaseFloat> targets = targets_reader.Value();
		Posterior cluster_targets = cluster_targets_reader.Value(utt);        
        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }

	    Vector<BaseFloat> utts;
        if (utt_weights != "") {
          utts = utt_weights_reader.Value(utt);
        } else {  // all per-frame weights are 1.0,
          utts.Resize(mat.NumRows());
          utts.Set(1.0);
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
	    cluster_targets_randomizer.AddData(cluster_targets);
        weights_randomizer.AddData(weights);
	    utt_randomizer.AddData(utts);
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
		utt_randomizer.Randomize(mask);
        cluster_targets_randomizer.Randomize(mask);
      }
      int32 batch_counter = 0;
      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next(),
										  cluster_targets_randomizer.Next(),
                                          utt_randomizer.Next()) {

        if(batch_counter < 5){
            batch_counter++;
            continue;
        }
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer.Value();
		const Posterior& cluster_nnet_tgt = cluster_targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();
        const Vector<BaseFloat>& utt_weights = utt_randomizer.Value();


        // forward pass
        nnet.Propagate(nnet_in, &nnet_out);
		regression_nnet.Propagate(nnet_out, &regression_nnet_out);
        cluster_nnet.Propagate(nnet_out, &cluster_nnet_out);

		xent.Eval(utt_weights, cluster_nnet_out, cluster_nnet_tgt, &cluster_obj_diff);
        mse.Eval(frm_weights, regression_nnet_out, nnet_tgt, &regression_obj_diff);
	     

        // backward pass
        if (!crossvalidate) {
          // backpropagate
          regression_in_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols());
          regression_nnet.Backpropagate(regression_obj_diff, &regression_in_diff);

          cluster_in_diff.Resize(nnet_out.NumRows(), nnet_out.NumCols());
          cluster_nnet.Backpropagate(cluster_obj_diff, &cluster_in_diff);

          regression_in_diff_copy = regression_in_diff;
          regression_in_diff.AddMat(error_ratio, cluster_in_diff);
            
          nnet.Backpropagate(regression_in_diff, NULL);
          //BP without update
          //regression_in_diff_copy.AddMat(error_ratio*error_step, cluster_in_diff);
          regression_in_diff_copy = cluster_in_diff;
          nnet.Feedbackward(regression_in_diff_copy,&nnet_in_diff);
          rand_in_diff.Resize(nnet_in.NumRows(),nnet_in.NumCols());
          rand_in_diff.SetRandn();
          //signum
          {
            nnet_in_diff.ApplyPowAbs(0,true);
            rand_in_diff.ApplyPowAbs(0,true);
          }

          int32 iMax = 10, jMax = 8;
          float mse_loss[iMax][jMax],xent_loss[iMax][jMax];
          for(int32 ii = 0; ii < iMax; ii++){
            for(int32 jj = 0 ; jj< jMax; jj++){

          nnet_in_copy = nnet_in;
          //add gradient
          nnet_in_copy.AddMat(grad_step*ii,nnet_in_diff);
          nnet_in_copy.AddMat(grad_step*jj,rand_in_diff);

          nnet.Propagate(nnet_in_copy,&nnet_out);
          regression_nnet.Propagate(nnet_out, &regression_nnet_out);
          cluster_nnet.Propagate(nnet_out, &cluster_nnet_out);

          mse_adv.Eval(frm_weights,regression_nnet_out,nnet_tgt,&regression_obj_diff);
		  xent_adv.Eval(utt_weights, cluster_nnet_out, cluster_nnet_tgt, &cluster_obj_diff);
          mse_loss[ii][jj] = mse_adv.BatchLoss();
          xent_loss[ii][jj] = xent_adv.BatchLoss();
                  
          }
          }

          KALDI_LOG << "Mse loss" ;
          for(int32 ii = 0; ii < iMax; ii++){
            for(int32 jj = 0 ; jj< jMax; jj++){
                KALDI_LOG << mse_loss[ii][jj];
            }
          }
          KALDI_LOG << "Xent loss" ;
          for(int32 ii = 0; ii < iMax; ii++){
            for(int32 jj = 0 ; jj< jMax; jj++){
                KALDI_LOG << xent_loss[ii][jj];
            }
          }
          KALDI_LOG << "Done.";
          return 0;
          regression_nnet.Backpropagate(regression_obj_diff, &regression_in_diff);
          cluster_nnet.Backpropagate(cluster_obj_diff, &cluster_in_diff);
          regression_in_diff.AddMat(error_ratio, cluster_in_diff);
          nnet.Backpropagate(regression_in_diff, NULL);
        }
        total_frames += nnet_in.NumRows();
      }
    }
    if (!crossvalidate) {
		 nnet.Write(target_model_filename, binary);
		 regression_nnet.Write(target_regression_model_filename, binary);
         cluster_nnet.Write(target_cluster_model_filename, binary);

    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  
	float total;
	//int a = 1; 
	if (!crossvalidate)
		total = 0.5* (mse.AvgLoss()+ mse_adv.AvgLoss())+ error_ratio * xent.AvgLoss();
	else
    		total = mse.AvgLoss() + error_ratio * xent.AvgLoss();	
	KALDI_LOG << mse.Report();
    KALDI_LOG << xent.Report();
    KALDI_LOG << mse_adv.Report();
	KALDI_LOG <<"Gatech " << mse.AvgLoss() << " " << xent.AvgLoss() << " " << mse_adv.Report();
	KALDI_LOG <<"LOG (nnet-train-ad[5.4.129~3-90363]:main():nnet-train-ad.cc:456) AvgLoss: " << total <<" (Mse+Xent)";
	KALDI_LOG <<"LOG (nnet-train-ad[5.4.129~3-90363]:main():nnet-train-ad.cc:456) BLCU: " << mse.AvgLoss() <<" (Mse)";
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
