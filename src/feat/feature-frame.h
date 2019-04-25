// feat/feature-spectrogram.h
// returns framed signal after windowing/pre-emphasis etc.

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

#ifndef KALDI_FEAT_FEATURE_FRAME_H_
#define KALDI_FEAT_FEATURE_FRAME_H_


#include <string>

#include "feat/feature-common.h"
#include "feat/feature-functions.h"
#include "feat/feature-window.h"

namespace kaldi {
/// @addtogroup  feat FeatureExtraction
/// @{


/// FrameOptions contains basic options for computing spectrogram
/// features.
struct FrameOptions {
  FrameExtractionOptions frame_opts;
  BaseFloat energy_floor;
  bool raw_energy;  // If true, compute energy before preemphasis and windowing

  FrameOptions() :
    energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
    raw_energy(true) {}

  void Register(OptionsItf *opts) {
    frame_opts.Register(opts);
    opts->Register("energy-floor", &energy_floor,
                   "Floor on energy (absolute, not relative) in Frame computation");
    opts->Register("raw-energy", &raw_energy,
                   "If true, compute energy before preemphasis and windowing");
  }
};

/// Class for computing spectrogram features.
class FrameComputer {
 public:
  typedef FrameOptions Options;
  explicit FrameComputer(const FrameOptions &opts);
  FrameComputer(const FrameComputer &other);

  const FrameExtractionOptions& GetFrameOptions() const {
    return opts_.frame_opts;
  }

  int32 Dim() const { return opts_.frame_opts.PaddedWindowSize() ; }

  bool NeedRawLogEnergy() { return opts_.raw_energy; }


  /**
     Function that computes one frame of spectrogram features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  This is ignored by this function, it's only
         needed for interface compatibility.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written.
  */
  void Compute(BaseFloat signal_log_energy,
               BaseFloat vtln_warp,
               VectorBase<BaseFloat> *signal_frame,
               VectorBase<BaseFloat> *feature);

  ~FrameComputer();

 private:
  FrameOptions opts_;
  BaseFloat log_energy_floor_;
  SplitRadixRealFft<BaseFloat> *srfft_;

  // Disallow assignment.
  FrameComputer &operator=(const FrameComputer &other);
};

typedef OfflineFeatureTpl<FrameComputer> Frame;


/// @} End of "addtogroup feat"
}  // namespace kaldi


#endif  // KALDI_FEAT_FEATURE_FRAME_H_
