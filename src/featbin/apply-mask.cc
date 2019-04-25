// featbin/reverse-feats.cc

// Copyright 2012 BUT, Mirko Hannemann

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
        "Apply learned irm to noisy feature to predict clean LPS\n"
        "Usage: reverse-feats [options] noisy-rspecifier mask-rspecifier out-wspecifier\n";
    ParseOptions po(usage);
    bool ignore_range = false;
    po.Register("ignore-range", &ignore_range, "if true, allow IRM to be beyond [0,1]");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string fspecifier = po.GetArg(1); //noisy lps
    std::string mspecifier = po.GetArg(2); //learned mask
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader feat_reader(fspecifier);
    SequentialBaseFloatMatrixReader mask_reader(mspecifier);
    int32 num_no_key = 0, num_bad_mask = 0,num_done=0;
    for (; !feat_reader.Done() && ! mask_reader.Done(); 
            feat_reader.Next(),mask_reader.Next()) {
      std::string key = feat_reader.Key();
      if(mask_reader.Key() != key){
        KALDI_WARN << "Missing key: " << key << " in noisy reader.";
        num_no_key++;
        continue;
      }
      Matrix<BaseFloat> feat = feat_reader.Value();
      Matrix<BaseFloat> mask = mask_reader.Value();
      KALDI_ASSERT(feat.NumRows() == mask.NumRows());
      KALDI_ASSERT(mask.NumCols() == feat.NumCols());

      if (feat.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
      if ( (!ignore_range) && (mask.Min() < 0 || mask.Max() > 1)){
        KALDI_WARN << "mask out of range of key " << key;
        num_bad_mask++;
        continue;
      }
      feat.ApplyExp();
      float floor = feat.Min()/10; //avoid log(0)
      mask.ApplyPow(2.0);
      feat.MulElements(mask);
      feat.ApplyFloor(floor);//in case mask has value < 0
      feat.ApplyLog();
      feat_writer.Write(key, feat);
      num_done++;
    }

    KALDI_LOG << "Done applying IRM for " << num_done << " files "
              << "with " << num_no_key << " missing key errors and"
              << "with " << num_bad_mask << " invalid mask errors.\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


