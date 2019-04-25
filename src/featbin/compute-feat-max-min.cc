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
      "Compute the global minimum and maximum for the feature files\n"  
	  "Usage: reverse-feats [options] in-rspecifier - -\n";
    ParseOptions po(usage);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
	float min_val, max_val;
	//std::vector<float> min_val_array,max_val_array;
    std::string rspecifier = po.GetArg(1);
    std::string wxpecifier_max = po.GetArg(2);
    std::string wxpecifier_min = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader(rspecifier);
	if(!feat_reader.Done()){
		min_val = feat_reader.Value().Min();
		max_val = feat_reader.Value().Max();
	}else
	  KALDI_ERR << "Could not initialize " << rspecifier;
	for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats  = feat_reader.Value();
      if (feats.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
	  min_val = (feats.Min()<min_val)?feats.Min():min_val;
	  max_val = (feats.Max()>max_val)?feats.Max():max_val;
	}

    Output ko_max(wxpecifier_max,false);
    ko_max.Stream() << max_val << "\t";
    Output ko_min(wxpecifier_min,false);
    ko_min.Stream() << min_val << "\n";
	    	
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


