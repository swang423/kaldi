// featbin/subsample-feats.cc

// Copyright 2012-2014  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <sstream>
#include <algorithm>
#include <iterator>
#include <utility>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Filter features by alignment. Use --remove=remove.txt\n"
        "to specify frames of corresponding alignments to be removed.\n"
        "Or use --keep=keep.txt to keep frames of corresponding alignments.\n"
        "Usage: filter-feats [options] <in-rspecifier> <ali-rspecifier> <out-wspecifier>\n"
        "  e.g. ali-to-pdf final.mdl ark:1.ali ark:- | filter-feats scp:feats.scp ark:- ark:-\n";

    ParseOptions po(usage);

    string remove = "", keep = "";
    po.Register("remove", &remove, "Kaldi object containing list of alignment to be filtered out.");
    po.Register("keep", &keep, "Kaldi object containing list of alignment to be kept.");
    string multicondition = "none";
    po.Register("multicondition", &multicondition, "If wsj, support wsj multicondition alignments.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3){
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT((remove == "" ) != (keep == "")); //one of them must be given

    string ali_rspecifier = po.GetArg(1);
    string rspecifier0 = po.GetArg(2);
    string wspecifier = po.GetArg(3);

    SequentialBaseFloatMatrixReader feat_reader0(rspecifier0);
    RandomAccessInt32VectorReader ali_reader(ali_rspecifier);
    BaseFloatMatrixWriter feat_writer(wspecifier);

    //Kaldi's vector does not support int32 container
    Vector<BaseFloat> vec_remove_host, vec_keep_host;
    std::vector<int32> vec_remove, vec_keep;
    if(remove != ""){
    ReadKaldiObject(remove, &vec_remove_host);
    vec_remove.resize(vec_remove_host.Dim());
      for (int32 jj = 0; jj<vec_remove_host.Dim();jj++){
        vec_remove[jj] = static_cast<int>(vec_remove_host(jj));
      }
    }
    if(keep != ""){
    ReadKaldiObject(keep, &vec_keep_host);
    vec_keep.resize(vec_keep_host.Dim());
      for (int32 jj = 0; jj<vec_keep_host.Dim();jj++){
        vec_keep[jj] = static_cast<int>(vec_keep_host(jj));
      }
    }

    int32 num_done = 0, num_no_ali = 0, num_len_mismatch=0;
    int64 frames_in = 0, frames_out = 0;

    // process all keys
    for (; !feat_reader0.Done(); feat_reader0.Next()) {
      string utt = feat_reader0.Key();
      const Matrix<BaseFloat> feats0(feat_reader0.Value());
    //alignment
      string utt_clean;
      if(multicondition == "wsj"){
        std::size_t pos_underscore = utt.find("_"); //for wsj
        KALDI_ASSERT(pos_underscore!=std::string::npos);
        utt_clean = utt.substr(0,pos_underscore);
      }else if(multicondition == "none"){
        utt_clean = utt;
      }else{
        KALDI_ERR << "Option " << multicondition << " is not accepted.";
      }
        
      if(!ali_reader.HasKey(utt_clean)){
        KALDI_WARN << "Missing alignment for utterance " << utt;
        num_no_ali++;
        continue;
      }
      const std::vector<int32> &ali = ali_reader.Value(utt_clean);
      if(ali.size() != feats0.NumRows()){
        KALDI_WARN << "Length mismatch of utt " << utt
        << " in alignment: " << feats0.NumRows() << " vs. " << ali.size() <<".";
        num_len_mismatch++;
        continue;
      }
        //row by row
        std::vector<bool> write_row(feats0.NumRows());

        int32 num_rows_to_keep;
        if(remove!=""){
          num_rows_to_keep = feats0.NumRows();
          for (int32 kk = 0 ; kk < feats0.NumRows(); kk++){ //iterate over frames
            write_row[kk] = true;
            for (int32 jj = 0; jj < vec_remove.size(); jj++){ //look for phonemes to remove
              if(vec_remove[jj] == ali[kk]){
                write_row[kk] = false;
                num_rows_to_keep--;
                break;
              }
            }
          } 
        }
        if(keep!=""){
          num_rows_to_keep = 0;
          for (int32 kk = 0 ; kk < feats0.NumRows(); kk++){ //iterate over frames
            write_row[kk] = false;
            for (int32 jj = 0; jj < vec_keep.size(); jj++){ //look for phonemes to remove
              if(vec_keep[jj] == ali[kk]){
                write_row[kk] = true;
                num_rows_to_keep++;
                break;
              }
            }
          } 
        }
        Matrix<BaseFloat> output0(num_rows_to_keep,feats0.NumCols());
        int32 jj =0;
        for (int32 kk = 0 ; kk < feats0.NumRows(); kk++){ //iterate over frames
            if(write_row[kk]){
                SubVector<BaseFloat> src0(feats0,kk),tgt0(output0,jj);
                tgt0.CopyFromVec(src0);
                jj++;
            }
        }
        KALDI_ASSERT(jj == num_rows_to_keep);
        feat_writer.Write(utt, output0);
        frames_in += feats0.NumRows();
        frames_out += num_rows_to_keep;
        num_done++;
    }
    KALDI_LOG << "Processed " << num_done << " feature matrices; " ;
    KALDI_LOG << "Processed " << frames_in << " input frames and "
              << frames_out << " output frames in " 
              << (remove=="" ? "keep" : "remove") << " mode.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
