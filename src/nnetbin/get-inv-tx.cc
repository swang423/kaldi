//sc011818

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
      "Get inverse transform (and apply global variance normalization)\n"
      "Usage:  get-inv-tx [options] <tx-in> <tx-out>\n"
      "e.g.:\n"
      " get-inv-tx --eta=2.0 tgt.fx inv.fx\n"
      " get-inv-tx --alpha=ark:alpha_gv_est.vec nnet.mdl nnet_txt.mdl\n";

    bool binary_write = false;
	float var_floor = 1e-10;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
	po.Register("var-floor", &var_floor,
        "Floor the variance, so the factors in <Rescale> are bounded.");
	float eta = 1; //See Yong Xu's global variance paper and nnet-compute-gv
	po.Register("eta",&eta, "Variance compensation factor in global variance normalization.");
	std::string alpha;
	po.Register("alpha",&alpha, "Kaldi object containing dimensional global variance.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
	KALDI_ASSERT(eta>0);
	if(alpha!="" && eta != 1){
	  KALDI_ERR << "Only one global variance parameter should be specified.";
	}
	Vector<BaseFloat> alpha_vec;
	if(alpha!=""){
	  ReadKaldiObject(alpha,&alpha_vec);
	}
    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load the network
    Nnet nnet, nnet1;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
			KALDI_ASSERT(nnet.NumComponents()==2);
    }
		AddShift& shift_comp = dynamic_cast<AddShift&>(nnet.GetComponent(0));
		Rescale& scale_comp = dynamic_cast<Rescale&>(nnet.GetComponent(1));
		Vector<BaseFloat> addshift_param(shift_comp.NumParams());
		Vector<BaseFloat> scale_param(scale_comp.NumParams());
		shift_comp.GetParams(&addshift_param);
		scale_comp.GetParams(&scale_param);
		scale_param.InvertElements();
		if(alpha!=""){
		  KALDI_ASSERT(alpha_vec.Dim()==scale_param.Dim());
		  scale_param.MulElements(alpha_vec);
		}else{
		  scale_param.Scale(eta);
		}
		for (int32 d = 0; d < scale_param.Dim(); d++) {
		if (scale_param(d) <= var_floor) {
		  KALDI_WARN << "Very small variance " << scale_param(d)
					 << " flooring to " << var_floor;
		  scale_param(d) = var_floor;
        }
		}
		addshift_param.Scale(-1);
		scale_comp.SetParams(scale_param);
		shift_comp.SetParams(addshift_param);
		
		nnet1.AppendComponent(scale_comp);
		nnet1.AppendComponent(shift_comp);
    // store the network,
    {
      Output ko(model_out_filename, binary_write);
      nnet1.Write(ko.Stream(), binary_write);
			KALDI_LOG << "Written inverse transform to: " << model_out_filename;
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
