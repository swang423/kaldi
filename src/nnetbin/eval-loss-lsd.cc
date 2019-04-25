// nnetbin/eval-loss.cc


#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


#ifndef M_LOG10E
#define M_LOG10E = 0.434294481903
#endif

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
      "Evaluate objective function of prediction and target.\n"
			"Writes a corresponding archive that maps utterance-id \n"
			"to loss of each uttererance, or print to stdout the overall \n"
			"average loss over all frames provided.\n"
      "Usage: eval-loss [options] <feature-rspecifier> <targets-rspecifier> <loss-wspecifier>|<loss-wxfilename>)\n"
      "e.g.: eval-loss ark:output.ark ark:target.ark -\n"
      "e.g.: eval-loss ark:output.ark ark:target.ark ark,t:loss.txt\n";

    ParseOptions po(usage);
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
        targets_rspecifier = po.GetArg(2),
		wspecifier_or_wxfilename = po.GetArg(3);


    kaldi::int64 tot_t = 0;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    Timer time;
    double time_now = 0;
    float cum_loss = 0;
    int32 num_done = 0;
	BaseFloatWriter average_loss_writer;
	bool write_ark = false;
    if (ClassifyWspecifier(wspecifier_or_wxfilename, NULL, NULL, NULL)
        != kNoWspecifier) {
        average_loss_writer.Open(wspecifier_or_wxfilename);
        write_ark = true;
    }
    // main loop,
    for (; !feature_reader.Done() && !targets_reader.Done(); feature_reader.Next(),targets_reader.Next()) {
        if(feature_reader.Key() != targets_reader.Key()){
            KALDI_ERR << "Mismatched utterance for feature " << feature_reader.Key() 
                << " and targets " << targets_reader.Key();
        }
      // read
      std::string utt = feature_reader.Key();
      Matrix<BaseFloat> mat = feature_reader.Value();
	Matrix<BaseFloat> targets = targets_reader.Value();
			KALDI_ASSERT(mat.NumRows() == targets.NumRows());
			KALDI_ASSERT(mat.NumCols() == targets.NumCols());
      KALDI_VLOG(2) << "Processing utterance " << num_done+1
                    << ", " << utt
                    << ", " << mat.NumRows() << " frm";

      if (!KALDI_ISFINITE(mat.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }
      if (!KALDI_ISFINITE(targets.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in targetss for " << utt;
      }
        //actual computation
        float ft_floor = mat.Max()-5*M_LN10;
        float tg_floor = targets.Max() - 5*M_LN10;
        mat.ApplyFloor(ft_floor);
        targets.ApplyFloor(tg_floor);
        mat.AddMat(-1.0,targets);
        mat.Scale(10*M_LOG10E);
        mat.ApplyPow(2);
        int32 feat_dim = mat.NumCols();
        int32 feat_row = mat.NumRows();
        Vector<BaseFloat> frame(feat_row);
        for (int32 kk = 0; kk<feat_row; kk++){
            float32 row_sum = std::sqrt(mat.Row(kk).Sum()/feat_dim);
            frame(kk) = row_sum;
        }
        cum_loss += frame.Sum();
        float average_loss_perutt = frame.Sum()/feat_row;
      	average_loss_writer.Write(utt, average_loss_perutt);

      // progress log,
      if (num_done % 1000 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    if(!write_ark){
        float average_loss = cum_loss/tot_t;

      Output ko(wspecifier_or_wxfilename, false); // text mode.
      ko.Stream() << "AvgLoss of " << num_done << " files: "
				<<  average_loss << "\n";
    }
 // final message,
    KALDI_LOG << "Done " << num_done << " files"
              << " in " << time.Elapsed()/60 << "min,"
              << " (fps " << tot_t/time.Elapsed() << ")";

#if HAVE_CUDA == 1
    if (GetVerboseLevel() >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
