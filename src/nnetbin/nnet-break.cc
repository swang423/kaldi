// nnetbin/nnet-concat.cc

// break nnet into a series of components
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-affine-transform.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
      "Write out nnet components"
      "Usage: nnet-concat [options] <nnet-in1> <...> <nnet-inN> <nnet-out>\n"
      "e.g.:\n"
      " nnet-concat --binary=false nnet.in nnet.out\n";

    ParseOptions po(usage);

    bool binary_write = false, mat = false;
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("mat", &mat, "Write out only mat. Only take affine");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1);
    std::string model_out_filename_base = po.GetArg(2);
    std::string model_out_filename;

    // read the first nnet,
    KALDI_LOG << "Reading " << model_in_filename;
    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    int32 num_componenets = nnet.NumComponents();
    // read all the other nnets,
    for (int32 i = 0; i < num_componenets; i++) {
      // read the nnet,
      model_out_filename = model_out_filename_base + "." + std::to_string(i);
      if(mat){
        Component::ComponentType comp_type = nnet.GetComponent(i).GetType();
        KALDI_LOG << "Component " << i << " : " << Component::TypeToMarker(comp_type);
        //bias is written to the first col
        //kaldi matrix is arranged as [#Outdim, #Indim]
        //nnet matrix is arrange as [#Indim, #Outdim]
        if(comp_type == Component::kAffineTransform){
            const AffineTransform& affine = dynamic_cast<AffineTransform&>(nnet.GetComponent(i));
            CuMatrix<BaseFloat> linearity_mt = CuMatrix<BaseFloat>(affine.GetLinearity());
            CuVector<BaseFloat> bias_mt = CuVector<BaseFloat>(affine.GetBias());
            Matrix<BaseFloat> linearity(linearity_mt.NumRows(),linearity_mt.NumCols()+1);
            linearity.ColRange(1,linearity_mt.NumCols()).CopyFromMat(linearity_mt);
            linearity.Transpose();
            linearity.Row(0).CopyFromVec(bias_mt);
            linearity.Transpose();
            WriteKaldiObject(linearity, model_out_filename, binary_write);
            KALDI_LOG << "Writing " << Component::TypeToMarker(comp_type) << " to " << model_out_filename;
        }else{
            KALDI_LOG << "Not writing " << Component::TypeToMarker(comp_type);
        }
      }else{
          Nnet nnet_next;
          nnet_next.AppendComponent(nnet.GetComponent(i));
          Output ko(model_out_filename, binary_write);
          nnet_next.Write(ko.Stream(), binary_write);
          KALDI_LOG << "Writing " << model_out_filename;
      }
    }

    // finally write the nnet to disk,

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


