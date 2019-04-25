// featbin/irm-post-process

// Copyright 2018, Sicheng Wang @ GT

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Use IRM to post-process (Yong Xu's IS15 paper)\n"
		"Usage: irm-post-process [options] nsy-lps-rspecifier pred-lps-rspecifier pred-irm-rspecifier out-wspecifier\n";
    ParseOptions po(usage);
	float upper_bound = 0.75;
	float lower_bound = 0.1;
	po.Register("upper-bound",&upper_bound, "if TF bin is higher than this value, use nsy");
	po.Register("lower-bound",&lower_bound, "if TF bin is lower than this value, use pred");

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

	KALDI_ASSERT(upper_bound<1);
	KALDI_ASSERT(lower_bound < upper_bound);
	KALDI_ASSERT(lower_bound>0);
    std::string nsy_rspecifier = po.GetArg(1),
				pred_rspecifier = po.GetArg(2),
				irm_rspecifier = po.GetArg(3);
    std::string wspecifier = po.GetArg(4);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader nsy_feat_reader(nsy_rspecifier);
    SequentialBaseFloatMatrixReader pred_feat_reader(pred_rspecifier);
    SequentialBaseFloatMatrixReader irm_feat_reader(irm_rspecifier);
	int32 num_error = 0;
    for (; !nsy_feat_reader.Done() && !pred_feat_reader.Done() && !irm_feat_reader.Done(); 
			nsy_feat_reader.Next(), pred_feat_reader.Next(), irm_feat_reader.Next()) {
      std::string key = nsy_feat_reader.Key();
	  if(key != pred_feat_reader.Key() || key != irm_feat_reader.Key()){
		KALDI_WARN << "Missing key for utt " << key;
		num_error++;
		continue;
	  }
      const Matrix<BaseFloat> &noisy  = nsy_feat_reader.Value();
      const Matrix<BaseFloat> &pred  = pred_feat_reader.Value();
      const Matrix<BaseFloat> &irm  = irm_feat_reader.Value();

      if (noisy.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
		num_error++;
        continue;
      }
	  KALDI_ASSERT(noisy.NumRows() == pred.NumRows());
	  KALDI_ASSERT(irm.NumRows() == pred.NumRows());
	  KALDI_ASSERT(noisy.NumCols() == pred.NumCols());
	  KALDI_ASSERT(irm.NumCols() == pred.NumCols());

      KALDI_ASSERT(irm.Min() >= 0);
      KALDI_ASSERT(irm.Max() <= 1);
      Matrix<BaseFloat> mean(noisy);
	  mean.AddMat(1.0,pred);
	  mean.Scale(0.5);
	  Matrix<BaseFloat> sa(irm), sb(irm),sc(irm);
	  sa.Add(-upper_bound);
	  sb = sa;
	  sa.ApplyHeaviside();//High SNR
	  sc.Scale(-1);
	  sc.Add(lower_bound);
	  sc.ApplyHeaviside();//Low SNR
	  sb.Set(1);
	  sb.AddMat(-1.0,sa);
	  sb.AddMat(-1.0,sc);//Medium SNR
	  sa.MulElements(noisy);
	  sb.MulElements(mean);
	  sc.MulElements(pred);
      sa.AddMat(1.0,sb);
	  sa.AddMat(1.0,sc);
	  feat_writer.Write(key, sa);
    }
	KALDI_LOG << "Done with " << num_error << "errors";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


