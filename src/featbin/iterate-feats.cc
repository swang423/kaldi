// featbin/paste-feats.cc
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
        "Useful for concatenating a list of small ark files each containing one feature into a large ark\n";
    
    ParseOptions po(usage);

    bool binary = true;
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
        != kNoRspecifier) {
      // We're operating on tables, e.g. archives.
      
      // Last argument is output
      string wspecifier = po.GetArg(po.NumArgs());
      BaseFloatMatrixWriter feat_writer(wspecifier);
    
      // Assemble vector of other input readers (with random-access)
      int32 num_done = 0, num_err = 0;
      for (int32 i = 1; i < po.NumArgs(); i++) {
        string rspecifier = po.GetArg(i);
        SequentialBaseFloatMatrixReader input(rspecifier);
        if(input.Done()){
            KALDI_WARN << "Empty ark " << rspecifier << " at index: " << i;
            num_err++;
            continue;
        }
        feat_writer.Write(input.Key(),input.Value());
        input.Close();//we expect only 1 feature per ark
        num_done++;
      }
      KALDI_LOG << "Done " << num_done << " utts, errors on "
                << num_err;

      return (num_done == 0 ? -1 : 0);
    } else {
        KALDI_ERR << "Not supporting arks." ;
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

