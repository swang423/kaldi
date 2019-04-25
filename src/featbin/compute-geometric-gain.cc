// It is equivalent to lossless mask (based on reconstruction) 
// mask = x.^2 / y.^2
// return 2 * log of gain
// so reconstruct by noisy lps + this mask

//Compute the gain defined in eq.19 of 
//https://ecs.utdallas.edu/loizou/speech/spcom_ga_june08.pdf
//to reconstruct, use eq. 4
//enhanced lps = noisy lps + 2log(gain)
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"

#ifndef M_SQ2INV
#define M_SQ2INV 0.7071067812
#endif

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute log of geometric gain\n"
        "Usage: compute-feats [options] clean-rspecifier noisy-rspecifier gain-wspecifier\n"
        "       compute-feats --predict=true gain-rspecifier noisy-rspecifier enh-wspecifier\n";
    ParseOptions po(usage);
      bool predict = false;
      po.Register("predict",&predict,"If true, give mask and noisy and return enhanced");
//    bool apply_log = true;
//    po.Register("apply-log", &apply_log,"If true, apply log to gain");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cspecifier = po.GetArg(1);
    std::string yspecifier = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader cln_reader(cspecifier);
    SequentialBaseFloatMatrixReader nsy_reader(yspecifier);
    int32 num_no_key = 0, num_done=0;
    for (; !cln_reader.Done() &&  !nsy_reader.Done();
            cln_reader.Next(),nsy_reader.Next()) {
      std::string key = cln_reader.Key();
      if(nsy_reader.Key() != key){
        KALDI_WARN << "Missing key: " << key << " in nsy reader.";
        num_no_key++;
        continue;
      }
      Matrix<BaseFloat> x = cln_reader.Value();
      const Matrix<BaseFloat>& y = nsy_reader.Value();
      if(predict){
        x.AddMat(1.0,y);
      }else{
        x.AddMat(-1.0,y);
      }
//      x.Scale(0.5);
      feat_writer.Write(key, x);
      num_done++;
    }

    KALDI_LOG << "Done calculating noise feats for " << num_done << " files "
              << "with " << num_no_key << " missing key errors.\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


