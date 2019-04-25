// featbin/reverse-feats.cc

//We follow the definition in https://arxiv.org/ftp/arxiv/papers/1709/1709.00917.pdf
//instead of https://ieeexplore.ieee.org/abstract/document/6639038 (without sqrt)

//Sicheng Wang @ GT
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
        "Given clean and noise (not noisy!) lps, output IRM\n"
        "Usage: reverse-feats [options] clean-rspecifier noise-rspecifier out-wspecifier\n";
    ParseOptions po(usage);
    std::string flooring = "none";
    float ibm_out = 0,
          lps_floor = 0;
    po.Register("ibm", &ibm_out,"If 1, output IBM mask. If greater than 1, then mask with higher threshold.");
    po.Register("lps-floor",&lps_floor, "If flooring == \"custom\", mask is only non-zero if clean lps > lps_floor");
    po.Register("flooring", &flooring, "[none:raw irm|mean use global clean lps mean as lps_floor |custom see lps-floor]");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cspecifier = po.GetArg(1);
    std::string nspecifier = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader cln_reader(cspecifier);
    SequentialBaseFloatMatrixReader nsy_reader(nspecifier);
    int32 num_no_key = 0, num_done=0;
    for (; !cln_reader.Done() && ! nsy_reader.Done(); 
            cln_reader.Next(),nsy_reader.Next()) {
      std::string key = cln_reader.Key();
      if(nsy_reader.Key() != key){
        KALDI_WARN << "Missing key: " << key << " in noisy reader.";
        num_no_key++;
        continue;
      }
      Matrix<BaseFloat> cln = cln_reader.Value();
      Matrix<BaseFloat> nsy = nsy_reader.Value();
      KALDI_ASSERT(cln.NumRows() == nsy.NumRows());
      KALDI_ASSERT(cln.NumCols() == nsy.NumCols());

      if (cln.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
      Matrix<BaseFloat> floor_mask;
      if (flooring == "mean" || flooring == "custom"){
        if (flooring == "mean"){
          lps_floor = cln.Sum()/(cln.NumRows() * cln.NumCols());
        }
        floor_mask = cln;
        floor_mask.Add(-1*lps_floor);
        floor_mask.ApplyHeaviside();
      }
//      cln.ApplyExp();
//      nsy.ApplyExp();
//      nsy.DivElements(cln);
      nsy.AddMat(-1.0,cln);
      nsy.ApplyExp();
      nsy.Add(1.0);
      nsy.ApplyPow(0.5);
      nsy.InvertElements();
      if (flooring == "mean" || flooring == "custom")
        nsy.MulElements(floor_mask);

      if (ibm_out >= 1){ // mask = 1 if Ps/Pn > threshold(ibm_out)
        float threshold = -1 * sqrt(ibm_out/(ibm_out + 1));
        nsy.Add(threshold);//If above, then speech
        nsy.ApplyHeaviside();
      }
      feat_writer.Write(key, nsy);
      num_done++;
    }

    KALDI_LOG << "Done calculating IRM for " << num_done << " files "
              << "with " << num_no_key << " missing key errors.\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


