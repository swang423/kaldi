// featbin/subset-feats.cc

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
        "Given the feats in the format computed by compute-framefft-feats\n"
        "Return the LPS minus log|0.5sin(2phi)|\n"
        "Except the first and last dim, which stays as LPS\n"
        "Usage: subset-feats [options] <in-rspecifier> <fft-wspecifier>\n"
        "Usage: subset-feats [options] <in-rspecifier> <lps-wspecifier> <phs-wspecifier>\n"
        "e.g.: subset-feats ark:- ark:lps.ark,ark:phs.ark\n";

    ParseOptions po(usage);

//    bool discretize_phase = true;
//    po.Register("discretize-phase", &discretize_phase, "If true, discretize phase to binary 0/1 first.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 ) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
   
    Vector<BaseFloat> lps_row, frame_row;
    Matrix<BaseFloat> lps_feats;
    for (; !kaldi_reader.Done(); kaldi_reader.Next()){
        const MatrixBase<BaseFloat> &frame_feats = kaldi_reader.Value();
        int32 num_frames = frame_feats.NumRows();
        int32 fft_dim_double = frame_feats.NumCols();
        KALDI_ASSERT(fft_dim_double % 4 == 0);
        int32 lps_dim = fft_dim_double/4+1;
        lps_feats.Resize(num_frames,lps_dim);
//        lps_row.Resize(lps_dim);
//        frame_row.Resize(fft_dim_double);
        for (int32 rr = 0; rr < num_frames ; rr ++){
/*            frame_row.CopyRowFromMat(frame_feats,rr);
            lps_row(0) = frame_row(0)*2; //Re(0) = 2logA
            lps_row(lps_dim-1) = frame_row(1)*2; // Re(N/2) = 2logA
            for (int32 cc = 1; cc < lps_dim-1; cc++){
                lps_row(cc) = frame_row(2*cc)+frame_row(2*cc+1); //Re(kk)+Im(kk) = 2logA+log|sin(2*phi)/2|
            }
            lps_feats.CopyRowFromVec(lps_row,rr);
*/
            lps_feats(rr,0) = frame_feats(rr,0)*2;
            lps_feats(rr,lps_dim-1) = frame_feats(rr,1)*2;
            for (int32 cc = 1; cc < lps_dim-1; cc++){
                lps_feats(rr,cc) = frame_feats(rr,2*cc)+frame_feats(rr,2*cc+1); //Re(kk)+Im(kk) = 2logA+log|sin(2*phi)/2|
            }
        }
        kaldi_writer.Write(kaldi_reader.Key(),lps_feats);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
