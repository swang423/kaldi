// feat/feature-spectrogram.cc

// Copyright 2009-2012  Karel Vesely
// Copyright 2012  Navdeep Jaitly

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


#include "feat/feature-framefft.h"


namespace kaldi {

FrameFftComputer::FrameFftComputer(const FrameFftOptions &opts)
    : opts_(opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  padded_window_size_ = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size_ & (padded_window_size_-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size_);
}

FrameFftComputer::FrameFftComputer(const FrameFftComputer &other):
    opts_(other.opts_), log_energy_floor_(other.log_energy_floor_), srfft_(NULL) {
  if (other.srfft_ != NULL)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*other.srfft_);
  padded_window_size_ = other.Dim();
}

FrameFftComputer::~FrameFftComputer() {
  delete srfft_;
}

void FrameFftComputer::Compute(BaseFloat signal_log_energy,
                                  BaseFloat vtln_warp,
                                  VectorBase<BaseFloat> *signal_frame,
                                  VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());


  // Compute energy after window function (not the raw one)
  if (!opts_.raw_energy)
    signal_log_energy = Log(std::max(VecVec(*signal_frame, *signal_frame),
                                     std::numeric_limits<BaseFloat>::epsilon()));

  if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
    srfft_->Compute(signal_frame->Data(), true);
  else  // An alternative algorithm that works for non-powers-of-two
    RealFft(signal_frame, true);

  //Arranged as Re(0),Re(N/2),Re(1),Im(1),...
  //IT IS IMPORTANT THAT THE WAV ARE READ AS INT16
  //NOT FLOAT. OTHERWISE THE DISTRIBUTION OF FEATURES 
  //WILL BE UNSATISFACTORY
  feature->CopyFromVec(*signal_frame);
  {
    feature->ApplyAbs();               //|x|
    feature->Add(1.0);                 //|x|+1
    feature->ApplyLog();               //log(|x|+1)
    signal_frame->ApplyPowAbs(0.0,true);     //sgn(x)
    feature->MulElements(*signal_frame);
  }

}

}  // namespace kaldi
