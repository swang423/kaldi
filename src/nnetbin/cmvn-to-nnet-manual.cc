// nnetbin/cmvn-to-nnet.cc

// Copyright 2012-2016  Brno University of Technology

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-various.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Specify mean and std \n"
      "Convert cmvn-stats into <AddShift> and <Rescale> components.\n"
      "Usage:  cmvn-to-nnet [options] <transf-in> <nnet-out>\n"
      "e.g.:\n"
      " cmvn-to-nnet --binary=false transf.mat nnet.mdl\n";


    bool binary_write = false;
    float std_dev = 1.0;
    float var_floor = 1e-10;
    float learn_rate_coef = 0.0;
    float mu = 0.0, sigma = 1.0;
    int32 num_dims = 257;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("std-dev", &std_dev, "Standard deviation of the output.");
    po.Register("var-floor", &var_floor,
        "Floor the variance, so the factors in <Rescale> are bounded.");
    po.Register("learn-rate-coef", &learn_rate_coef,
        "Initialize learning-rate coefficient to a value.");
    po.Register("mu",&mu,"mean");
    po.Register("sigma",&sigma,"std");
    po.Register("dim",&num_dims,"dimension");
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_out_filename = po.GetArg(1);

    // buffers for shift and scale
    Vector<BaseFloat> shift(num_dims);
    Vector<BaseFloat> scale(num_dims);
    shift.Set(-mu);
    scale.Set(std_dev/sigma);
    // create empty nnet,
    Nnet nnet;

    // append shift component to nnet,
    {
      AddShift shift_component(shift.Dim(), shift.Dim());
      shift_component.SetParams(shift);
      shift_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(shift_component);
    }

    // append scale component to nnet,
    {
      Rescale scale_component(scale.Dim(), scale.Dim());
      scale_component.SetParams(scale);
      scale_component.SetLearnRateCoef(learn_rate_coef);
      nnet.AppendComponent(scale_component);
    }

    // write the nnet,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
      KALDI_LOG << "Written cmvn in 'nnet1' model to: " << model_out_filename;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
