#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
		"Given feat-to-len output, write target spk id that many times\n"
		"The utt2spk_rspecifier must be 0 based\n"
        "Usage: len-to-spkid   [options] <feats-len-rspecifier> <utt2spk-rspecifier><pdfs-wspecifier>\n"
        "e.g.: \n"
        " ali-to-pdf 1.mdl ark:1.ali ark, t:-\n";
    ParseOptions po(usage);
    int32 offset = 0;
    po.Register("offset", &offset, "if non-zero, then add offset to spk id");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string featlen_rspecifier = po.GetArg(1),
		utt2spk_rspecifier = po.GetArg(2),
        pdfs_wspecifier = po.GetArg(3);

	SequentialInt32Reader featlen_reader(featlen_rspecifier);
	RandomAccessInt32Reader utt2spk_reader(utt2spk_rspecifier);
    Int32VectorWriter writer(pdfs_wspecifier);

    int32 num_done = 0,num_miss_uttspk=0;
    for (; !featlen_reader.Done(); featlen_reader.Next()) {
      std::string key = featlen_reader.Key();
	  if(!utt2spk_reader.HasKey(key)){
		KALDI_WARN << "Missing " << key << "in utt2spk.";
		num_miss_uttspk++;
		continue;
	  }
	  int32 num_frame = featlen_reader.Value();
	  int32 spk_id = utt2spk_reader.Value(key);

      std::vector<int32> alignment(num_frame);

      for (size_t i = 0; i < num_frame; i++)
        alignment[i] = spk_id + offset;

      writer.Write(key, alignment);
      num_done++;
    }
    KALDI_LOG << "Converted " << num_done << " alignments to pdf sequences;\n" 
			  << "with " << num_miss_uttspk << " missing utt2spk.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


