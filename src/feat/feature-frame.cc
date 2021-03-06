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


#include "feat/feature-frame.h"


namespace kaldi {

FrameComputer::FrameComputer(const FrameOptions &opts)
    : opts_(opts), srfft_(NULL) {
  if (opts.energy_floor > 0.0)
    log_energy_floor_ = Log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

FrameComputer::FrameComputer(const FrameComputer &other):
    opts_(other.opts_), log_energy_floor_(other.log_energy_floor_), srfft_(NULL) {
  if (other.srfft_ != NULL)
    srfft_ = new SplitRadixRealFft<BaseFloat>(*other.srfft_);
}

FrameComputer::~FrameComputer() {
  delete srfft_;
}

void FrameComputer::Compute(BaseFloat signal_log_energy,
                                  BaseFloat vtln_warp,
                                  VectorBase<BaseFloat> *signal_frame,
                                  VectorBase<BaseFloat> *feature) {
  KALDI_ASSERT(signal_frame->Dim() == opts_.frame_opts.PaddedWindowSize() &&
               feature->Dim() == this->Dim());
//  feature->Resize(signal_frame->Dim());
  feature->CopyFromVec(*signal_frame);
//  Vector<BaseFloat> tmp(*signal_frame);
    
}

}  // namespace kaldi
