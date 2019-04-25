// nnetbin/nnet-copy.cc


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-parallel-component.h"
#include "nnet/nnet-linear-transform.h"
#include "nnet/nnet-affine-transform.h"
#include "cudamatrix/cu-device.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-various.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    const char *usage =
			"NOT COMPLETE\n"
      "Copy Neural Network model (and possibly change binary/text format)\n"
      "Usage:  nnet-copy [options] <model-in> <model-out>\n"
      "e.g.:\n"
      " nnet-copy --binary=false nnet.mdl nnet_txt.mdl\n";

    bool binary_write = true;
    int32 remove_first_components = 0;
    int32 remove_last_components = 0;
    BaseFloat dropout_rate = -1.0;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");

    po.Register("remove-first-layers", &remove_first_components,
        "Deprecated, please use --remove-first-components");
    po.Register("remove-last-layers", &remove_last_components,
        "Deprecated, please use --remove-last-components");

    po.Register("remove-first-components", &remove_first_components,
        "Remove N first Components from the Nnet");
    po.Register("remove-last-components", &remove_last_components,
        "Remove N last layers Components from the Nnet");

    po.Register("dropout-rate", &dropout_rate,
        "Probability that neuron is dropped"
        "(-1.0 keeps original value).");

    std::string from_parallel_component;
    po.Register("from-parallel-component", &from_parallel_component,
        "Extract nested network from parallel component (two possibilities: "
        "'3' = search for ParallelComponent and get its 3rd network; "
        "'1:3' = get 3nd network from 1st component; ID = 1..N).");
	int32 st_to_mt_dim = 0;
	po.Register("st-to-mt-dim",&st_to_mt_dim, "Assume the output layer is affine! Add auxiliary output of specified dimension to nnet, assuming from 1 task up to 2.");
	int32 mt_to_st_dim = 0;
	po.Register("mt-to-st-dim",&mt_to_st_dim, "Remove auxiliary output from nnet, assuming from 2 tasks down to 1.");
	bool st_to_mt_dup = false;
	po.Register("st-to-mt-dup",&st_to_mt_dup, "If true, duplicate output with identity matrix");
	bool set_linearity_to_identity = false;
	po.Register("set-linearity-to-identity",&set_linearity_to_identity, "Set all affine to identity and bias to zero.");
	float temperature = 1;
	po.Register("temperature",&temperature,"Insert a Rescale layer before Softmax for soft target with temperature;\nSee Hinton's knowledge distillation.");
    int32 st_to_mt_splice = 1;
    po.Register("st-to-mt-splice",&st_to_mt_splice,"duplicate outputs with splice");
    int32 mt_to_st_splice = 1;
    po.Register("mt-to-st-splice",&mt_to_st_splice,"extract outputs with splice");
    bool add_lin,add_lhn;
    po.Register("add-lin", &add_lin, "add linear input layer");
    po.Register("add-lhn", &add_lin, "add linear hidden layer before last non-linear");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
	KALDI_ASSERT(temperature>0);
    std::string model_in_filename = po.GetArg(1),
        model_out_filename = po.GetArg(2);

    // load the network
    Nnet nnet;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
    }

    // eventually replace 'nnet' by nested network from <ParallelComponent>,
    if (from_parallel_component != "") {
      std::vector<int32> component_id_nested_id;
      kaldi::SplitStringToIntegers(from_parallel_component, ":", false,
                                   &component_id_nested_id);
      // parse the argument,
      int32 component_id = -1, nested_id = 0;
      switch (component_id_nested_id.size()) {
        case 1:
          nested_id = component_id_nested_id[0];
          break;
        case 2:
          component_id = component_id_nested_id[0];
          nested_id = component_id_nested_id[1];
          break;
        default:
          KALDI_ERR << "Check the csl '--from-parallel-component='"
                    << from_parallel_component
                    << " There must be 1 or 2 elements.";
      }
      // search for first <ParallelComponent> (we don't know component_id yet),
      if (component_id == -1) {
        for (int32 i = 0; i < nnet.NumComponents(); i++) {
          if (nnet.GetComponent(i).GetType() == Component::kParallelComponent) {
            component_id = i+1;
            break;
          }
        }
      }
      // replace the nnet,
      KALDI_ASSERT(nnet.GetComponent(component_id-1).GetType() ==
                   Component::kParallelComponent);
      ParallelComponent& parallel_comp =
        dynamic_cast<ParallelComponent&>(nnet.GetComponent(component_id-1));
      nnet = parallel_comp.GetNestedNnet(nested_id-1);  // replace!
    }

    // optionally remove N first components,
    if (remove_first_components > 0) {
      for (int32 i = 0; i < remove_first_components; i++) {
        nnet.RemoveComponent(0);
      }
    }

    // optionally remove N last components,
    if (remove_last_components > 0) {
      for (int32 i = 0; i < remove_last_components; i++) {
        nnet.RemoveLastComponent();
      }
    }
	// set all affine to unity
	if (set_linearity_to_identity) {
	  for (int32 i = 0; i < nnet.NumComponents(); i++) {
		if (nnet.GetComponent(i).GetType() == Component::kAffineTransform) {
		  AffineTransform* new_affine = dynamic_cast<AffineTransform*>(&nnet.GetComponent(i));
		  Matrix<BaseFloat> new_linearity(new_affine->GetLinearity().NumRows(),new_affine->GetLinearity().NumCols());
		  Vector<BaseFloat> new_bias(new_affine->GetBias().Dim());
		  new_linearity.SetUnit();
		  new_bias.SetZero();
		  new_affine->SetLinearity(CuMatrix<BaseFloat>(new_linearity));
		  new_affine->SetBias(CuVector<BaseFloat>(new_bias));
		  AffineTransform new_affine_instance = *new_affine;
		  nnet.ReplaceComponent(i,new_affine_instance);
		  //nnet.ReplaceComponent(i,*new_affine);
		}
	  }
	}
    if(mt_to_st_splice>1){
	  int32 num_comp = nnet.NumComponents();
	  KALDI_ASSERT(nnet.GetComponent(num_comp-1).GetType() == Component::kAffineTransform);
      int32 all_context = mt_to_st_splice * 2 + 1;
	  const AffineTransform& affine = dynamic_cast<AffineTransform&>(nnet.GetComponent(num_comp-1)); 
	  CuMatrix<BaseFloat> linearity_mt = CuMatrix<BaseFloat>(affine.GetLinearity());
	  CuVector<BaseFloat> bias_mt = CuVector<BaseFloat>(affine.GetBias());
      KALDI_ASSERT(bias_mt.Dim()%all_context==0);
      int32 dim_single = bias_mt.Dim()/all_context;
//        KALDI_LOG << "dim single: " << dim_single;
	  CuMatrix<BaseFloat> linearity_st(dim_single,linearity_mt.NumCols());
	  CuVector<BaseFloat> bias_st(dim_single);
//        KALDI_LOG << "linearity: " << mt_to_st_splice << " " ;
                  
      linearity_st.CopyFromMat(linearity_mt.RowRange(mt_to_st_splice*dim_single,dim_single));
      bias_st.CopyFromVec(bias_mt.Range(mt_to_st_splice*dim_single,dim_single));
	  AffineTransform* affine_st = new AffineTransform(linearity_st.NumCols(),dim_single);
	  affine_st->SetLinearity(linearity_st);
	  affine_st->SetBias(bias_st);
	  nnet.ReplaceComponent(num_comp-1,*affine_st); 
    }
    //make output splice
    if(st_to_mt_splice>1){
	  int32 num_comp = nnet.NumComponents();
	  KALDI_ASSERT(nnet.GetComponent(num_comp-1).GetType() == Component::kAffineTransform);
	  const AffineTransform& affine = dynamic_cast<AffineTransform&>(nnet.GetComponent(num_comp-1)); 
	  CuMatrix<BaseFloat> linearity_st = CuMatrix<BaseFloat>(affine.GetLinearity());
	  CuVector<BaseFloat> bias_st = CuVector<BaseFloat>(affine.GetBias());
	  CuMatrix<BaseFloat> linearity_mt(linearity_st.NumRows()*st_to_mt_splice,linearity_st.NumCols());
	  CuVector<BaseFloat> bias_mt(bias_st.Dim()*st_to_mt_splice);
      for(int32 kk = 0; kk<st_to_mt_splice; kk++){
        linearity_mt.RowRange(kk*linearity_st.NumRows(),linearity_st.NumRows()).CopyFromMat(linearity_st);
        bias_mt.Range(kk*bias_st.Dim(),bias_st.Dim()).CopyFromVec(bias_st);
      }
	  AffineTransform* affine_mt = new AffineTransform(linearity_st.NumCols(),linearity_st.NumRows()*st_to_mt_splice);
	  affine_mt->SetLinearity(linearity_mt);
	  affine_mt->SetBias(bias_mt);
	  nnet.ReplaceComponent(num_comp-1,*affine_mt); 
        
    }
    //create mt, init with rand
	if(st_to_mt_dim >0){
	  int32 num_comp = nnet.NumComponents();
//	  if(nnet.GetComponent(num_comp-1).GetType() == Component::kAffineTransform){//regression network 
	  KALDI_ASSERT(nnet.GetComponent(num_comp-1).GetType() == Component::kAffineTransform);
	  const AffineTransform& affine = dynamic_cast<AffineTransform&>(nnet.GetComponent(num_comp-1)); 
	  CuMatrix<BaseFloat> linearity_st = CuMatrix<BaseFloat>(affine.GetLinearity());
	  CuVector<BaseFloat> bias_st = CuVector<BaseFloat>(affine.GetBias());
	  CuMatrix<BaseFloat> linearity_mt(linearity_st.NumRows()+st_to_mt_dim,linearity_st.NumCols());
	  CuVector<BaseFloat> bias_mt(bias_st.Dim()+st_to_mt_dim);
	  linearity_mt.RowRange(0,linearity_st.NumRows()).CopyFromMat(linearity_st);
	  linearity_mt.RowRange(linearity_st.NumRows(),st_to_mt_dim).SetRandn();
	  bias_mt.Range(0,bias_st.Dim()).CopyFromVec(bias_st);
	  bias_mt.Range(bias_st.Dim(),st_to_mt_dim).SetRandn();
	  
	  AffineTransform* affine_mt = new AffineTransform(affine.InputDim(),linearity_st.NumRows()+st_to_mt_dim);
	  affine_mt->SetLinearity(linearity_mt);
	  affine_mt->SetBias(bias_mt);
	  nnet.ReplaceComponent(num_comp-1,*affine_mt); 
//	  }
	  //For classifaction networks ending with softmax layers, the suggestion is to remove it first, split last affine, then use block softmax 
	}
	if (mt_to_st_dim > 0){
			
	  KALDI_ASSERT(nnet.GetLastComponent().GetType() == Component::kAffineTransform);
	  Component* last_component = nnet.GetLastComponent().Copy();
		AffineTransform* last_component_pt = dynamic_cast<AffineTransform*>(last_component);
		CuMatrix<BaseFloat> last_lin;
		last_lin = last_component_pt->GetLinearity();
		CuVector<BaseFloat> last_bias;
		last_bias = last_component_pt->GetBias();
		CuMatrix<BaseFloat> new_last_lin;
		CuVector<BaseFloat> new_last_bias;
		int32 output_dim = last_lin.NumRows();
//		KALDI_ASSERT(output_dim%2==0);
//		KALDI_ASSERT(last_bias.Dim()==output_dim);
		new_last_lin.Resize(output_dim - mt_to_st_dim,last_lin.NumCols());
		new_last_bias.Resize(output_dim - mt_to_st_dim);
		new_last_lin.CopyFromMat(last_lin.RowRange(0,output_dim-mt_to_st_dim));
		new_last_bias.CopyFromVec(last_bias.Range(0,output_dim - mt_to_st_dim));
		AffineTransform new_last_component = AffineTransform(last_lin.NumCols(),output_dim - mt_to_st_dim);
		new_last_component.SetLinearity(new_last_lin);
		new_last_component.SetBias(new_last_bias);
		nnet.ReplaceComponent(nnet.NumComponents()-1,new_last_component);
	}
	//By default, we duplicate from 1 task to 2
	if(st_to_mt_dup){
	  LinearTransform split_component = LinearTransform(nnet.OutputDim(),nnet.OutputDim()*2);
	  Matrix<BaseFloat> split_mat(nnet.OutputDim()*2,nnet.OutputDim());
	  split_mat.RowRange(0,nnet.OutputDim()).SetUnit();
	  split_mat.RowRange(nnet.OutputDim(),nnet.OutputDim()).SetUnit();
	  split_component.SetLinearity(CuMatrix<BaseFloat>(split_mat));
      split_component.SetLearnRateCoef(0); //trainble = false
      //split_component.SetBiasLearnRateCoef(0); //linear has no bias
	  nnet.AppendComponent(split_component);
	}
    // dropout,
    if (dropout_rate != -1.0) {
      nnet.SetDropoutRate(dropout_rate);
    }
	if (temperature > 1){
//	  int32 num_comp = nnet.NumComponents();
//	  KALDI_ASSERT(nnet.GetComponent(num_comp-1).GetType() == Component::kSoftmax);
//	  KALDI_ASSERT(components_[affine_index[i]]->GetType()==Component::kAffineTransform)
//	  const Softmax& softmax = dynamic_cast<Softmax&>(nnet.GetComponent(num_comp-1)); 
	  Component* last_component = nnet.GetLastComponent().Copy();
	  Rescale scale(last_component->InputDim(),last_component->InputDim());
	  Vector<BaseFloat> scale_vec(last_component->InputDim());
	  scale_vec.Set(1.0/temperature);
	  scale.SetParams(scale_vec);
	  scale.SetLearnRateCoef(0);
	  nnet.RemoveLastComponent();
	  nnet.AppendComponent(scale);
	  nnet.AppendComponent(*last_component);	  
	}else if(temperature<1){
	  KALDI_WARN << "Temperature < 1 is not advised.";
	}
    if (add_lin){

    }
    if (add_lhn){
        KALDI_ERR << "NOT IMPLEMENTED";
    }
    // store the network,
    {
      Output ko(model_out_filename, binary_write);
      nnet.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written 'nnet1' to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
