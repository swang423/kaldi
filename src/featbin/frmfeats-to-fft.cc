// feats_in should be output of compute-framefft-feats
// For int16 wav (not float), compute-framefft-feats computes
// feats y=sgn(x).*log(|x|+1) where x is the 512 dim RFFT
// Hence this program computes the reverse
// x = sgn(y).*[exp(|y|)-1]
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
        "Return the LPS and phase.\n"
        "Usage: subset-feats [options] <in-rspecifier> <fft-wspecifier>\n"
        "Usage: subset-feats [options] <in-rspecifier> <lps-wspecifier> <phs-wspecifier>\n"
        "e.g.: subset-feats ark:- ark:lps.ark,ark:phs.ark\n";

    ParseOptions po(usage);

//    bool discretize_phase = true;
//    po.Register("discretize-phase", &discretize_phase, "If true, discretize phase to binary 0/1 first.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2 && po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    std::string phs_wspecifier;

    BaseFloatMatrixWriter mag_writer(wspecifier);
    BaseFloatMatrixWriter phs_writer;
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    if (po.NumArgs() == 3){
        phs_wspecifier = po.GetArg(3);
        phs_writer.Open(phs_wspecifier);
    }
    Vector<BaseFloat> mag_frame,phs_frame;
    Matrix<BaseFloat> mag_feats,phs_feats,combine_feats;
    for (; !kaldi_reader.Done(); kaldi_reader.Next()){
        const MatrixBase<BaseFloat>& feats = kaldi_reader.Value();
        int32 num_frames = feats.NumRows();
        int32 fft_dim = feats.NumCols();
        KALDI_ASSERT(fft_dim % 2 == 0);
        int32 fft_dim_half = fft_dim/2+1;
        mag_feats.Resize(num_frames,fft_dim_half);
        phs_feats.Resize(num_frames,fft_dim_half);
        mag_frame.Resize(fft_dim);
        phs_frame.Resize(fft_dim);

        Matrix<BaseFloat> sign_feats(feats);
        Matrix<BaseFloat> frame_feats(feats);
        sign_feats.ApplyPowAbs(0.0,true);           //sgn(y)
        frame_feats.ApplyPowAbs(1.0,false);         //|y|
        frame_feats.ApplyExp();                     //exp(|y|)
        KALDI_ASSERT(KALDI_ISFINITE(frame_feats.Sum()));
        frame_feats.Add(-1.0);                      //exp(|y|)-1
        frame_feats.MulElements(sign_feats);
        for (int32 kk = 0 ; kk < num_frames; kk ++){
            mag_frame.CopyRowFromMat(frame_feats,kk);
            phs_frame.CopyRowFromMat(frame_feats,kk);
            ComputePowerSpectrum(&mag_frame);
            ComputePhaseSpectrum(&phs_frame);
            mag_feats.CopyRowFromVec(mag_frame.Range(0,fft_dim_half),kk);
            phs_feats.CopyRowFromVec(phs_frame.Range(0,fft_dim_half),kk);
        }
        mag_feats.ApplyFloor(std::numeric_limits<BaseFloat>::epsilon());
        mag_feats.ApplyLog();
        if(po.NumArgs()==2){
            combine_feats.Resize(num_frames,fft_dim_half*2);
            combine_feats.ColRange(0,fft_dim_half).CopyFromMat(mag_feats);
            combine_feats.ColRange(fft_dim_half,fft_dim_half).CopyFromMat(phs_feats);
            mag_writer.Write(kaldi_reader.Key(),combine_feats);
        }else{
            mag_writer.Write(kaldi_reader.Key(),mag_feats);
            phs_writer.Write(kaldi_reader.Key(),phs_feats);
        }
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
