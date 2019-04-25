//Author: Sicheng Wang @ GT
//4.24.2019
//sichengwang.gatech@gmail.com

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"
#include "feat/wave-reader.h"

namespace kaldi {
    void ComputeFftSpectrum(SubVector<BaseFloat> lps_,SubVector<BaseFloat> phs_,Vector<BaseFloat
> *fft_) {
    //Inverse of ComputePowerSpectrum
    //it's stored as [real0, realN/2-1, real1, im1, real2, im2, ...]
        KALDI_ASSERT(lps_.Dim() == phs_.Dim());
        int32 fft_dim_ = (lps_.Dim()-1)*2;
        lps_.Scale(0.5);
        lps_.ApplyExp();
        KALDI_ASSERT(KALDI_ISFINITE(lps_.Sum()));
        fft_->Resize(fft_dim_);
        (*fft_)(0) = lps_(0) * cos(phs_(0));
        (*fft_)(1) = lps_(lps_.Dim()-1) * cos(phs_(phs_.Dim()-1));
        for (int32 i = 1 ; i < lps_.Dim()-1; i++){
            (*fft_)(2*i) = lps_(i) * cos(phs_(i));
            (*fft_)(2*i+1) = lps_(i) * sin(phs_(i));
        }
    }
    void RoundToInt(Vector<BaseFloat> *x){
        for (int32 i = 0 ; i < x->Dim(); i++){
            (*x)(i) = floor((*x)(i));
        }
    }
    void GetHammingWindow(std::string type, int32 size, Vector<BaseFloat> *w){
        if(type == "rectangular"){
          for (int32 i = 0; i < size; i++)
            (*w)(i) = 1.0;
        }else if(type == "hamming") {
          double a = M_2PI / (size-1);
          for (int32 i = 0; i < size; i++) {
            double i_fl = static_cast<double>(i);
            (*w)(i) = 0.54 - 0.46*cos(a * i_fl);
          }
        }else{
          KALDI_ERR << "Other windows are not allowed\n"
            << "You may implemented easily by looking at feat/feature-window.cc";
        }
    }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Wave reconstruction\n"
        "If no output is given, we will write each individual wav out to disk specified\n"
        "at output_dir at Fs = 16k . Otherwise, we write to ark/scp and you may use \n"
        "wav-copy <rxspecifier> <wxspecifier> to extract individual wav later\n"
        "Usage: overlap-add --output-dir=/tmp <lps-rspecifier> <phs-rspecifier> \n"
        "Usage: overlap-add <lps-rspecifier> <phs-rspecifier> <wav-wspecifier>\n"
        "       overlap-add scp:lps.scp scp:phs.scp ark:tmp.ark\n"
        "       wav-copy tmp.ark:24 tmp_0.wav\n";

    ParseOptions po(usage);
      BaseFloat fs = 16000;
      po.Register("fs",&fs,"Sampling frequency");
      int32 window_shift = 256, fft_size = 512;
      po.Register("window-shift",&window_shift, "window shift in samples");
      po.Register("fft-size", &fft_size, "inverse FFT size");
      std::string window_type = "hamming",
                  output_dir = "./";
      po.Register("window-type", &window_type, "Type of window (\"hamming\") only");
      po.Register("output-dir", &output_dir, "Output directory");
/*
       int32 WindowShift() const {
          return static_cast<int32>(samp_freq * 0.001 * frame_shift_ms);
        }
*/
    po.Read(argc, argv);
    bool out_is_rspecifier;
    if ( po.NumArgs() == 3) {
      out_is_rspecifier = true;
    }else if (po.NumArgs() == 2){
      out_is_rspecifier = false;
    }else{
      po.PrintUsage();
      exit(1);
    }
    // Set-up

    /// For historical reasons, we scale waveforms to the range
    /// (2^15-1)*[-1, 1], not the usual default DSP range [-1, 1].
    const BaseFloat kWaveSampleMax = 32767.0, kWaveSampleMin = -32768;

    SplitRadixRealFft<BaseFloat> *srfft;
    if ((fft_size & (fft_size-1)) == 0) //power of two
      srfft = new SplitRadixRealFft<BaseFloat>(fft_size);
    else
      KALDI_ERR << "We only support FFT size of power of 2.";

    Vector<BaseFloat> window(fft_size);
    GetHammingWindow(window_type,fft_size,&window);
    Vector<BaseFloat> window_sq(window);
    window_sq.ApplyPow(2.0);

    std::string mspecifier = po.GetArg(1);  //magnitude
    std::string pspecifier = po.GetArg(2);  //phase
    SequentialBaseFloatMatrixReader lps_reader(mspecifier);
    SequentialBaseFloatMatrixReader phs_reader(pspecifier);

    std::string wspecifier;
    TableWriter<WaveHolder> wav_writer;
    if(out_is_rspecifier){
        wspecifier = po.GetArg(3);  //wav
        wav_writer.Open(wspecifier);
    }
    int32 num_no_key = 0, num_done=0;

    Vector<BaseFloat> frame_fft;
    for (; !lps_reader.Done() &&  !phs_reader.Done();
            lps_reader.Next(),phs_reader.Next()) {
      std::string key = lps_reader.Key();
      if(phs_reader.Key() != key){
        KALDI_WARN << "Missing key: " << key << " in phase reader.";
        num_no_key++;
        continue;
      }
      const Matrix<BaseFloat>& lps = lps_reader.Value();
      const Matrix<BaseFloat>& phs = phs_reader.Value();
      int32 num_frames = lps.NumRows(), lps_dim = lps.NumCols();
      KALDI_ASSERT(num_frames == phs.NumRows());
      KALDI_ASSERT(lps_dim == phs.NumCols());
      //init
      int32 signal_length = fft_size + (num_frames-1)*window_shift;
      Vector<BaseFloat> wav_vector(signal_length),wav_denominator(signal_length);
      wav_vector.Set(0);
      wav_denominator.Set(0);
      //OLA
      for (int32 r = 0; r < num_frames; r++){
        ComputeFftSpectrum(lps.Row(r),phs.Row(r),&frame_fft);
        srfft->Compute(frame_fft.Data(),false);//irfft
        frame_fft.Scale((float)(1)/(float)(fft_size));
        frame_fft.MulElements(window);
        int32 idx_begin = r*window_shift;
        wav_vector.Range(idx_begin,fft_size).AddVec(1.0,frame_fft);
        wav_denominator.Range(idx_begin,fft_size).AddVec(1.0,window_sq);
      }
      wav_vector.DivElements(wav_denominator);
      RoundToInt(&wav_vector);
      if(wav_vector.Max() > kWaveSampleMax){
        KALDI_WARN << "Maximum " << wav_vector.Max() << " exceeds ceiling in utt "
                   << key;
        wav_vector.ApplyCeiling(kWaveSampleMax);
      }
      if(wav_vector.Min() < kWaveSampleMin){
        KALDI_WARN << "Minimum " << wav_vector.Min() << " exceeds floor in utt "
                   << key;
        wav_vector.ApplyFloor(kWaveSampleMin);
      }
      //write
      Matrix<BaseFloat> wav_matrix(1,signal_length);
      wav_matrix.CopyRowsFromVec(wav_vector);
      WaveData output(fs, wav_matrix);
      if(out_is_rspecifier)
        wav_writer.Write(key, output);
      else{
        std::string wav_out_fn = output_dir+"/"+key+".wav";
        Output ko(wav_out_fn, true, false);
        if (!WaveHolder::Write(ko.Stream(), true, output)) {
            KALDI_ERR << "Write failure to " << wav_out_fn;
        }
      }
      num_done++;
    }
    KALDI_LOG << "Done reconstructing wavs for " << num_done << " files "
              << "with " << num_no_key << " missing key errors.\n";
    if(!out_is_rspecifier){
      KALDI_LOG << "Output wav written to " << output_dir;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

