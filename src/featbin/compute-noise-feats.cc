//This is semi-reverse of compute-irm-feats
//This computes noise lps from clean lps and the basic irm
//We follow the definition in https://arxiv.org/ftp/arxiv/papers/1709/1709.00917.pdf
//instead of https://ieeexplore.ieee.org/abstract/document/6639038 (without sqrt)
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
        "Given clean lps and irm, output noise\n"
        "Usage: reverse-feats [options] clean-rspecifier noise-rspecifier out-wspecifier\n";
    ParseOptions po(usage);
    float mask_floor = 1e-12;
    po.Register("mask-floor", &mask_floor,"To prevent 1/0 and log(0)");

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cspecifier = po.GetArg(1);
    std::string mspecifier = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter feat_writer(wspecifier);
    SequentialBaseFloatMatrixReader cln_reader(cspecifier);
    SequentialBaseFloatMatrixReader mask_reader(mspecifier);
    int32 num_no_key = 0, num_done=0;
    for (; !cln_reader.Done() && ! mask_reader.Done(); 
            cln_reader.Next(),mask_reader.Next()) {
      std::string key = cln_reader.Key();
      if(mask_reader.Key() != key){
        KALDI_WARN << "Missing key: " << key << " in mask reader.";
        num_no_key++;
        continue;
      }
      Matrix<BaseFloat> cln = cln_reader.Value();
      Matrix<BaseFloat> msk = mask_reader.Value();
      KALDI_ASSERT(cln.NumRows() == msk.NumRows());
      KALDI_ASSERT(cln.NumCols() == msk.NumCols());

      if (cln.NumRows() == 0) {
        KALDI_WARN << "Empty feature matrix for key " << key;
        continue;
      }
      KALDI_ASSERT(msk.Min()>0 && msk.Max()<=1);
      msk.ApplyFloor(mask_floor);
      msk.InvertElements();
      msk.ApplyPow(2.0);
      msk.Add(-1.0);
//      KALDI_ASSERT(msk.Min()>0);//before log
      msk.ApplyFloor(mask_floor);
      msk.ApplyLog();
      msk.AddMat(1.0,cln);
      KALDI_ASSERT(KALDI_ISFINITE(msk.Sum()));
      feat_writer.Write(key, msk);
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


