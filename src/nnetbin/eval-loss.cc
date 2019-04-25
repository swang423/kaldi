// nnetbin/eval-loss.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
//#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
      "Evaluate objective function of prediction and target.\n"
			"Writes a corresponding archive that maps utterance-id \n"
			"to loss of each uttererance, or print to stdout the overall \n"
			"average loss over all utterances provided.\n"
      "Usage: eval-loss [options] <feature-rspecifier> <targets-rspecifier> <loss-wspecifier>|<loss-wxfilename>)\n"
      "e.g.: eval-loss ark:output.ark ark:target.ark -\n"
      "e.g.: eval-loss ark:output.ark ark:target.ark ark,t:loss.txt\n";

    ParseOptions po(usage);

		std::string objective_function = "mse";
		po.Register("objective-function", &objective_function,
				"Use objective function (xent|mse|cosine|multitask)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in front of main network (in nnet format)");

    std::string targets_transform;
    po.Register("targets-transform", &targets_transform, 
				"Targets transform in Nnet format");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");
    
    LossOptions loss_opts;
    loss_opts.Register(&po);

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        targets_rspecifier = po.GetArg(2),
				wspecifier_or_wxfilename = po.GetArg(3);

    // Select the GPU
#if HAVE_CUDA == 1
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

    kaldi::int64 tot_t = 0;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    CuMatrix<BaseFloat> feats_transf, targets_transf, obj_diff, obj_diff_perutt;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;
		Xent xent(loss_opts);
        Mse mse(loss_opts);
    MultiTaskLoss multitask(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      multitask.InitFromString(objective_function);
    }
        
		BaseFloatWriter average_loss_writer;
		bool write_ark = false;
		if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
			!= kNoWspecifier) {
			average_loss_writer.Open(wspecifier_or_wxfilename);
			write_ark = true;
		}
    // main loop,
    for (; !feature_reader.Done() && !targets_reader.Done(); feature_reader.Next(),targets_reader.Next()) {
			if(feature_reader.Key() != targets_reader.Key()){
				KALDI_ERR << "Mismatched utterance for feature " << feature_reader.Key() 
					<< " and targets " << targets_reader.Key();
			}
      // read
      std::string utt = feature_reader.Key();
      CuMatrix<BaseFloat> mat = CuMatrix<BaseFloat>(feature_reader.Value());
			CuMatrix<BaseFloat> targets = CuMatrix<BaseFloat>(targets_reader.Value());
			KALDI_ASSERT(mat.NumRows() == targets.NumRows());
			KALDI_ASSERT(mat.NumCols() == targets.NumCols());
      KALDI_VLOG(2) << "Processing utterance " << num_done+1
                    << ", " << utt
                    << ", " << mat.NumRows() << " frm";

      if (!KALDI_ISFINITE(mat.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }
      if (!KALDI_ISFINITE(targets.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in targetss for " << utt;
      }
      // fwd-pass, feature transform,
      nnet_transf.Feedforward(mat, &feats_transf);
      if (!KALDI_ISFINITE(feats_transf.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      }
      // fwd-pass, targets transform,
      nnet_transt.Feedforward(targets, &targets_transf);
      if (!KALDI_ISFINITE(targets_transf.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-targets for " << utt;
      }
        //use unit weights
        Vector<BaseFloat> weights;
      weights.Resize(mat.NumRows());
      weights.Set(1.0);
    Xent xent_perutt(loss_opts);
    Mse mse_perutt(loss_opts);
    MultiTaskLoss multitask_perutt(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      multitask_perutt.InitFromString(objective_function);
    }
            
			//evaluate objective function we've chosen
			if (objective_function == "xent") {
				// gradients re-scaled by weights in Eval,
				if(write_ark)
					xent_perutt.Eval(weights, feats_transf, targets_transf, &obj_diff_perutt);
				else
					xent.Eval(weights, feats_transf, targets_transf,&obj_diff);
			} else if (objective_function == "mse") {
				// gradients re-scaled by weights in Eval,
                if(write_ark)
                    mse_perutt.Eval(weights, feats_transf,targets_transf, &obj_diff_perutt);
                else
                    mse.Eval(weights, feats_transf,targets_transf, &obj_diff);
            } else if (0 == objective_function.compare(0, 9, "multitask")) {
                if(write_ark)
                    multitask_perutt.Eval(weights, feats_transf,targets_transf, &obj_diff_perutt);
                else
                    multitask.Eval(weights, feats_transf,targets_transf, &obj_diff);
			} else {
				KALDI_ERR << "Unknown objective function code : " << objective_function;
			}

			if(write_ark){
				BaseFloat average_loss_perutt;
				if (objective_function == "xent") {
					average_loss_perutt = xent_perutt.AvgLoss();
				} else if (objective_function == "mse") {
					average_loss_perutt = mse_perutt.AvgLoss();
                } else if (0 == objective_function.compare(0, 9, "multitask")) {
                    average_loss_perutt = multitask_perutt.AvgLoss();
				} else {
					KALDI_ERR << "Unknown objective function code : " << objective_function;
				}
				KALDI_ASSERT(KALDI_ISFINITE(obj_diff_perutt.Sum()));
      	average_loss_writer.Write(utt, average_loss_perutt);
			}
      // progress log,
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
		if(!write_ark){
			BaseFloat average_loss;
			if (objective_function == "xent") {
				average_loss = xent.AvgLoss();
				KALDI_LOG << xent.Report();
				KALDI_LOG << xent.ReportPerClass();
			} else if (objective_function == "mse") {
				average_loss = mse.AvgLoss();
				KALDI_LOG << mse.Report();
            } else if (0 == objective_function.compare(0, 9, "multitask")) {
                average_loss = multitask.AvgLoss();
                KALDI_LOG << multitask.Report();
			} else {
				KALDI_ERR << "Unknown objective function code : " << objective_function;
			}

      Output ko(wspecifier_or_wxfilename, false); // text mode.
      ko.Stream() << "AvgLoss of " << num_done << " files: "
				<<  average_loss << "\n";
    }
 // final message,
    KALDI_LOG << "Done " << num_done << " files"
              << " in " << time.Elapsed()/60 << "min,"
              << " (fps " << tot_t/time.Elapsed() << ")";

#if HAVE_CUDA == 1
    if (GetVerboseLevel() >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
