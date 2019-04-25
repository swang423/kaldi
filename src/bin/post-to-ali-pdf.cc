// bin/ali-to-post.cc

// Copyright 2009-2012  Microsoft Corporation, Go-Vivace Inc.,
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/posterior.h"

namespace kaldi {

void PosteriorToAlignment(const Posterior &post,bool shift,std::vector<int32> *ali){
  ali->clear();
  ali->resize(post.size());
  for (size_t i = 0; i < post.size(); i++) {
    KALDI_ASSERT(post[i].size()==1);
    KALDI_ASSERT(post[i][0].second == 1.0);
    KALDI_ASSERT( (shift && post[i][0].first > 0) || ((!shift)&& post[i][0].first >=0) );
    if(shift)
        (*ali)[i] = post[i][0].first - 1;
    else
        (*ali)[i] = post[i][0].first;
  }
}
}
/** @brief Convert alignments to viterbi style posteriors. The aligned
    symbol gets a weight of 1.0 */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert posteriors to alignments.\n"
        "Default will shift 1-based post to 0-based pdf.\n"
        "Usage:  ali-to-post [options] <post-rspecifier> <ali-wspecifier>\n"
        "e.g.:\n"
        " ali-to-post ark:1.post ark:1.ali\n"
        "See also: ali-to-pdf, ali-to-phones, show-alignments, post-to-weights\n";
    ParseOptions po(usage);
    bool shift_base = true;
    po.Register("shift-base",&shift_base,"If true, shift 1-based post to 0-based pdf");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string posteriors_rspecifier = po.GetArg(1);
    std::string alignments_wspecifier = po.GetArg(2);

    int32 num_done = 0;
    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);
    Int32VectorWriter alignment_writer(alignments_wspecifier);

    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      num_done++;
      const Posterior &post = posterior_reader.Value();
      std::vector<int32> alignment;
      PosteriorToAlignment(post,shift_base,&alignment);
      alignment_writer.Write(posterior_reader.Key(), alignment);
    }
    KALDI_LOG << "Converted " << num_done << " posterior.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

