// featbin/copy-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"

#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458
#endif

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Copy features [and possibly change format]\n"
        "Usage: copy-feats [options] <feature-rspecifier> <feature-wspecifier>\n"
        "or:   copy-feats [options] <feats-rxfilename> <feats-wxfilename>\n"
        "e.g.: copy-feats ark:- ark,scp:foo.ark,foo.scp\n"
        " or: copy-feats ark:foo.ark ark,t:txt.ark\n"
        "See also: copy-matrix, copy-feats-to-htk, copy-feats-to-sphinx, select-feats,\n"
        "extract-rows, subset-feats, subsample-feats, splice-feats, paste-feats,\n"
        "concat-feats\n";

    ParseOptions po(usage);
    bool binary = true;
		bool apply_exp = false;
		bool apply_log = false;
    std::string num_frames_wspecifier;
    po.Register("binary", &binary, "Binary-mode output (not relevant if writing "
                "to archive)");
    po.Register("write-num-frames", &num_frames_wspecifier,
                "Wspecifier to write length in frames of each utterance. "
                "e.g. 'ark,t:utt2num_frames'.  Only applicable if writing tables, "
                "not when this program is writing individual files.  See also "
                "feat-to-len.");
		po.Register("apply-exp",&apply_exp, "apply exponential to all features,"
								"Also subtract the exponential by 30*ln(2),"
								"This option should be exclusive with other math operations.");
		po.Register("apply-log",&apply_log, "apply logarithm to all features"
								"Also add the exponential by 30*ln(2) before log,"
								"This option should be exclusive with other math operations.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;
		if(apply_exp && apply_log)
			KALDI_ERR << "Only one math operation is allowed.";

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) != kNoRspecifier) {
      // Copying tables of features.
      std::string rspecifier = po.GetArg(1);
      std::string wspecifier = po.GetArg(2);
      Int32Writer num_frames_writer(num_frames_wspecifier);

        BaseFloatMatrixWriter kaldi_writer(wspecifier);
          SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
					Matrix<BaseFloat> feat_matrix;
          for (; !kaldi_reader.Done(); kaldi_reader.Next(), num_done++) {
						feat_matrix = kaldi_reader.Value();
						if(apply_exp){
							feat_matrix.Add(-30 * M_LN2);
							feat_matrix.ApplyExp();
							KALDI_ASSERT(KALDI_ISFINITE(feat_matrix.Sum()));
						}
						if(apply_log){
							feat_matrix.ApplyLog();
							feat_matrix.Add(30 * M_LN2);
							KALDI_ASSERT(KALDI_ISFINITE(feat_matrix.Sum()));
						}
						
            kaldi_writer.Write(kaldi_reader.Key(), feat_matrix);
            if (!num_frames_wspecifier.empty())
              num_frames_writer.Write(kaldi_reader.Key(),
                                      kaldi_reader.Value().NumRows());
          }
      KALDI_LOG << "Copied " << num_done << " feature matrices.";
      return (num_done != 0 ? 0 : 1);
    } else {
      if (!num_frames_wspecifier.empty())
        KALDI_ERR << "--write-num-frames option not supported when writing/reading "
                  << "single files.";

      std::string feat_rxfilename = po.GetArg(1), feat_wxfilename = po.GetArg(2);

      Matrix<BaseFloat> feat_matrix;
      ReadKaldiObject(feat_rxfilename, &feat_matrix);
			if(apply_exp){
				feat_matrix.Add(-30 * M_LN2);
				feat_matrix.ApplyExp();
				KALDI_ASSERT(KALDI_ISFINITE(feat_matrix.Sum()));
			}
			if(apply_log){
				feat_matrix.ApplyLog();
				feat_matrix.Add(30 * M_LN2);
				KALDI_ASSERT(KALDI_ISFINITE(feat_matrix.Sum()));
			}
      WriteKaldiObject(feat_matrix, feat_wxfilename, binary);
      KALDI_LOG << "Modified features from " << PrintableRxfilename(feat_rxfilename)
                << " to " << PrintableWxfilename(feat_wxfilename);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
