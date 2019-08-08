// featbin/transform-feats.cc

// Given enh nnet prediction and asr posterior
// Select frames with high posterior
// Add phone-level residue to enhancement feats
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Usage: residue-feats [options] <residue-rxfilename> <enh-feats-rspecifier> <post-mat-rspecifier> <post-processed-wspecifier>\n"
        "Example: residue-feats residue.txt ark:enh_out.ark ark:asr_out.ark ark:post_process.ark\n";
        
    ParseOptions po(usage);
    BaseFloat post_threshold=0.6;
    po.Register("post-threshold", &post_threshold, "Only add residue if phone-level posterior is higher than this value");
    std::string multicondition = "none";
    po.Register("multicondition", &multicondition, "If wsj, then use wsj-style multicondition for posterior");
    std::string model_name = "", to_phoneme = "";
    po.Register("model-name",&model_name,"mdl file to convert senone to monophone");
    po.Register("to-phoneme", &to_phoneme, "If given, convert monophone to phoneme. See ali-to-phone");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string residue_rxfilename = po.GetArg(1);
    std::string feat_rspecifier = po.GetArg(2);
    std::string post_rspecifier = po.GetArg(3);
    std::string feat_wspecifier = po.GetArg(4);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    RandomAccessBaseFloatMatrixReader post_reader(post_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);

    Matrix<BaseFloat> residue;
    ReadKaldiObject(residue_rxfilename, &residue);

    TransitionModel trans_model;
    if(model_name !=""){
        ReadKaldiObject(model_name, &trans_model);
    }
    Vector<BaseFloat> lookup_table;
    if (to_phoneme != ""){
        ReadKaldiObject(to_phoneme, &lookup_table);
    }
    std::string utt_clean;
    int32 num_done = 0, num_missing = 0;
    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      Matrix<BaseFloat> feat = feat_reader.Value();
      KALDI_ASSERT(feat.NumCols() == residue.NumCols());
      if(multicondition == "wsj"){
		std::size_t pos_underscore = utt.find("_"); //for wsj
		KALDI_ASSERT(pos_underscore!=std::string::npos);
		utt_clean = utt.substr(0,pos_underscore);
      }else if(multicondition == "none"){
		utt_clean = utt;
      }else{
        KALDI_ERR << "Option " << multicondition << " is not accepted.";
      }
      if(!post_reader.HasKey(utt_clean)){
        KALDI_WARN << "Missing key for utt: " << utt << ". Not processed.";
        num_missing++;
      }else{
          const Matrix<BaseFloat> &post_mat = post_reader.Value(utt_clean);
          KALDI_ASSERT(post_mat.NumRows() == feat.NumRows());
          int32 num_skip = 0;
          for(int32 row = 0; row < post_mat.NumRows(); row++){
            SubVector<BaseFloat> post_vec = post_mat.Row(row);
            int32 max_index;
            BaseFloat max_value = post_vec.Max(&max_index);
            if(max_value < post_threshold){
              num_skip++;
              continue;
            }
            if(model_name != "")
              max_index = trans_model.TransitionIdToPhone(max_index);
            if(to_phoneme != "")
              max_index = (int32)lookup_table(max_index);
            KALDI_ASSERT(max_index < residue.NumRows());
//            KALDI_VLOG(2) << "   Max Phone: " << max_index << "(" << max_value << ") at row " << row;
            const SubVector<BaseFloat> residue_vec = residue.Row(max_index);
            feat.Row(row).AddVec(1.0,residue_vec);
            if(row > 4)
                break;
          }
          KALDI_VLOG(3) << "Utt " << utt << ": skipped " << num_skip << " out of " << post_mat.NumRows() << " frames.";
      }
      feat_writer.Write(utt, feat);
      num_done++;
    }
    KALDI_LOG << "Added residue to " << num_done << " utterances and "  << num_missing << " miss alignments.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
