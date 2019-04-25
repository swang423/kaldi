// featbin/subset-feats.cc
//x_post = (x_noisy - gamma) .* sigmoid(x_pred,alpha,beta)
//where sigmoid(x,a,b) = [1+exp(-a*(x-b*u))]^(-1) where u is the mean of x

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "feat/feature-common.h"
#include "feat/feature-functions.h"
#include "feat/feature-window.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Given mean normalized feats, this outputs an ideal ratio mask\n"
        "There are two modes of operation depending on the number of arguments\n"
        "If two, then read enhanced feats and outputs mask corresponds to that feature\n"
        "If three, then read enhanced and noisy feats, and outputs post processed enhanced feature\n"
        "Usage: subset-feats [options] <in-rspecifier> <mask-wspecifier>\n"
        "Usage: subset-feats [options] <enh-rspecifier> <nsy-rspecifier> <enh-wspecifier>\n"
        "e.g.: subset-feats ark:- ark:lps.ark,ark:phs.ark\n";

    ParseOptions po(usage);

    bool linear = false,
         online_mean = false;
    po.Register("linear",&linear,"If true, apply mask in linear instead of log domain");
    po.Register("online-mean",&online_mean,"If true, the input feature can be unnormalized and will use utterance mean to normalize");
    BaseFloat floor = -20, alpha=1,beta=1,gamma=0,delta=0;
    po.Register("floor",&floor,"The minimum in input feats will be floored to this value");
    po.Register("alpha", &alpha, "Growth rate of sigmoid");
    po.Register("beta", &beta, "beta*feature mean will be the offset in sigmoid");
    po.Register("gamma", &gamma, "gamma will be the offset to be subtracted from noisy feature");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(po.NumArgs());
    std::string noisy_rspecifier;

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    SequentialBaseFloatMatrixReader noisy_reader;
    if (po.NumArgs() == 3){
        noisy_rspecifier = po.GetArg(2);
        noisy_reader.Open(noisy_rspecifier);
    }
    Matrix<BaseFloat> noisy_feats;
    for (; !kaldi_reader.Done(); kaldi_reader.Next()){
        if(noisy_reader.IsOpen() && noisy_reader.Done())
            break;
        const MatrixBase<BaseFloat> &frame_feats = kaldi_reader.Value();
        if(noisy_reader.IsOpen()){
            KALDI_ASSERT(kaldi_reader.Key() == noisy_reader.Key());
            noisy_feats = noisy_reader.Value();
            KALDI_ASSERT(frame_feats.NumRows() == noisy_feats.NumRows());
            KALDI_ASSERT(frame_feats.NumCols() == noisy_feats.NumCols());
        }
        Matrix<BaseFloat> feats(frame_feats);
        if(online_mean){
            BaseFloat mean = feats.Sum()/(feats.NumRows()*feats.NumCols());
            feats.Add(-beta*mean);
        }
        //logistic sigmoid (1+exp(-alpha*(x-beta*mean)))^(-1)
        //We want to safeguard against numerical instability 
        {
            feats.Scale(-alpha);//-x*alpha
            feats.ApplyCeiling(-floor);
            feats.ApplyExp();//exp(-x)
            KALDI_ASSERT(KALDI_ISFINITE(feats.Sum()));
            feats.Add(1.0);//1+exp(-x)
            feats.InvertElements();// (1+exp(-x))^(-1)
        }
        KALDI_ASSERT(KALDI_ISFINITE(feats.Sum()));
        if(noisy_reader.IsOpen()){
            if(linear)
                noisy_feats.ApplyExp();
            noisy_feats.Add(gamma);
            feats.MulElements(noisy_feats);
            if(linear)
                feats.ApplyLog();
        }
        kaldi_writer.Write(kaldi_reader.Key(),feats);
        if(noisy_reader.IsOpen())
            noisy_reader.Next();
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
