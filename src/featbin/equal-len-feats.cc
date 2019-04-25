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

namespace kaldi {

void MatchLenFrames(const Matrix<BaseFloat> &in,
                   Matrix<BaseFloat> *out) {
    if(out->NumRows() > in.NumRows()){
        int32 in_row = in.NumRows();
        int32 repeat = out->NumRows() / in_row;
        for(int32 kk = 0 ; kk < repeat ; kk++){
            out->RowRange(kk*in_row,in_row).CopyFromMat(in);
        }
        int32 diff = out->NumRows() - repeat*in_row;
        KALDI_ASSERT(diff<in_row);
        if(diff>0){
            out->RowRange(repeat*in_row,diff).CopyFromMat(in.RowRange(0,diff));
        }
    }else if (out->NumRows() < in.NumRows()){
        (*out) = in.RowRange(0,out->NumRows());
    }else{ 
        (*out) = in;
    }
}

}  // namespace kaldi,

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Reverse features in time (for backwards decoding)\n"
        "Usage: reverse-feats [options] ref-rspecifier unmatched-rspecifier out-wspecifier\n";
    ParseOptions po(usage);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_rspecifier = po.GetArg(1);
    std::string tgt_rspecifier = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader ref_feat_reader(ref_rspecifier);
    SequentialBaseFloatMatrixReader tgt_feat_reader(tgt_rspecifier);
    for (; !ref_feat_reader.Done() && !tgt_feat_reader.Done(); ref_feat_reader.Next(),
                                                               tgt_feat_reader.Next()) {
      std::string key = ref_feat_reader.Key();
      KALDI_ASSERT(key == tgt_feat_reader.Key());
      const Matrix<BaseFloat> &ref_feats  = ref_feat_reader.Value();
      const Matrix<BaseFloat> &tgt_feats  = tgt_feat_reader.Value();
      KALDI_ASSERT(ref_feats.NumCols() == tgt_feats.NumCols());

      if (ref_feats.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }

      Matrix<BaseFloat> new_feats(ref_feats);
      MatchLenFrames(tgt_feats, &new_feats);
      feat_writer.Write(key, new_feats);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


