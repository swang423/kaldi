
#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "hmm/posterior.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform one iteration of Neural Network linear output adaptation by stochastic gradient descent.\n"
        "Usage:  nnet-adapt-lhn-kl [options] <model-in> <feature-rspecifier> <alignments-rspecifier> [<model-out>] [<model-si>]\n"
        "e.g.: \n"
        " nnet-adapt-lhn-kl nnet.init scp:train.scp ark:train.ali nnet.iter1 nnet.si\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
	NnetDataRandomizerOptions rnd_opts;
	rnd_opts.Register(&po);
    LossOptions loss_opts;
    loss_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

//    int32 bunchsize=512, cachesize=32768, seed=777;
//    po.Register("bunchsize", &bunchsize, "Size of weight update block");
//    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling (max 8388479)");
//    po.Register("seed", &seed, "Seed value for srand, sets fixed order of frame-shuffling");
   
    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    kaldi::int32 max_frames = 6000; // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
	std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,"Objective function : xent|multitask");
	bool multi_condition_post = false;
	po.Register("multi-condition-post",&multi_condition_post, "If true, use multi-condition post. See nnet-train-frmshuff-multi-condition");    
    
    po.Read(argc, argv);

    if (po.NumArgs() != 6-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posterior_rspecifier = po.GetArg(3),
		mean_rspecifier = po.GetArg(4),
        pre_rspecifier = po.GetArg(5);
//        soft_rspecifier = po.GetArg(4),
        
    std::string target_model_filename, spkind_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(6);
//      spkind_model_filename = po.GetArg(8);
    }

     
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
//    CuDevice::Instantiate().DisableCaching();
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);
   
/*    Nnet sinnet;
    if (!crossvalidate) {
    sinnet.Read(spkind_model_filename);
    sinnet.SetTrainOptions(trn_opts);
	}
*/
    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posterior_reader(posterior_rspecifier);
//    SequentialBaseFloatMatrixReader soft_reader(soft_rspecifier);
    //RandomAccessBaseFloatMatrixReader soft_reader(soft_rspecifier);
	RandomizerMask randomizer_mask(rnd_opts);
	MatrixRandomizer feature_randomizer(rnd_opts);
//	MatrixRandomizer soft_randomizer(rnd_opts);
	PosteriorRandomizer targets_randomizer(rnd_opts);
	VectorRandomizer weights_randomizer(rnd_opts);

	Xent xent(loss_opts);
	MultiTaskLoss multitask(loss_opts);
    if (0 == objective_function.compare(0, 9, "multitask")) {
      multitask.InitFromString(objective_function);
    }

    
    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, obj_diff, mean, pre;
    Posterior targets;
	Vector<BaseFloat> frm_weights(rnd_opts.minibatch_size);
	frm_weights.Set(1);
    Matrix<BaseFloat> pre_host, mean_host;
    bool binary_in;
    Input ki(mean_rspecifier, &binary_in);
    mean_host.Read(ki.Stream(), binary_in);
    mean.Resize(mean_host.NumRows(), mean_host.NumCols());
    mean.CopyFromMat(mean_host);

    Input ki_pre(pre_rspecifier, &binary_in);
    pre_host.Read(ki_pre.Stream(), binary_in);
    pre.Resize(pre_host.NumRows(), pre_host.NumCols());
    pre.CopyFromMat(pre_host);
    Timer time;
    double time_now = 0;
    double time_next = 0;
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_alignment = 0, num_other_error = 0;
	while (!feature_reader.Done() ) {
#if HAVE_CUDA==1
  CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the cache
      for (;!feature_reader.Done() ; feature_reader.Next()) {
		if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key();
		std::string utt_post;
        KALDI_VLOG(3) << "Reading utt " << utt;
        // check that we have alignments
		if(multi_condition_post){
		  std::size_t pos_underscore = utt.find("_");
		  KALDI_ASSERT(pos_underscore!=std::string::npos);
		  utt_post = utt.substr(0,pos_underscore);
		  if(!posterior_reader.HasKey(utt)) {
			if (posterior_reader.HasKey(utt_post)){
			  utt = utt_post;
			}
			else{
			  KALDI_WARN << utt_post << ", missing targets";
			  num_no_alignment++;
			  continue;
			}
		  }
		}else{
		  if (!posterior_reader.HasKey(utt)) {
			KALDI_WARN << utt << ", missing targets";
			num_no_alignment++;
			continue;
		  }
		}
/*		//if (soft_reader.HasKey(utt)){
		if (soft_reader.Key()!=utt){
		  KALDI_WARN << utt << ", missing soft targets";
		  KALDI_LOG << "rspec: " << soft_rspecifier << "; " << soft_reader.IsOpen();
		  num_no_soft++;
		  continue;
		}*/
        // get feature alignment pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Posterior post = posterior_reader.Value(utt);
//		Matrix<BaseFloat> soft_post = soft_reader.Value();
//		KALDI_LOG << "DEBUG: read soft_post done for: " << utt;
//		post_dim = soft_post.NumCols();
//		KALDI_ASSERT(post_dim*2 == nnet.OutputDim());
        // check maximum length of utterance
        if (mat.NumRows() > max_frames) {
          KALDI_WARN << "Utterance " << utt << ": Skipped because it has " << mat.NumRows() 
					<< " frames, which is more than " << max_frames << ".";
          num_other_error++;
          continue;
        }
        // check length match of features/alignments
        if ((int32)post.size() != mat.NumRows()) {
          int32 diff = post.size() -  mat.NumRows();
          int32 tolerance = 5; // allow some tolerance (truncate)
          if (diff > 0 && diff < tolerance) { // alignment longer
            for(int32 i=0; i<diff; i++) { post.pop_back(); }
          }
          if (diff < 0 && abs(diff) < tolerance) { // feature matrix longer
            for(int32 i=0; i<abs(diff); i++) { mat.RemoveRow(mat.NumRows()-1); }
          }
          // check again
          if ((int32)post.size() != mat.NumRows()) {
            KALDI_WARN << "Length mismatch of alignment "<< (post.size()) << " vs. features "<< (mat.NumRows());
            num_other_error++;
            feature_reader.Next();
            continue;
          }
        }
/*        {
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back((int32)post.size());
          lenght.push_back(soft_post.NumRows());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if((int32)post.size() != min) post.resize(min);
            if(soft_post.NumRows() != min) soft_post.Resize(min, soft_post.NumRows(),kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << post.size()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }*/
        // check max value in alignment corresponds to NN output
//        KALDI_ASSERT(*std::max_element(alignment.begin(),alignment.end()) < nnet.OutputDim());
        // possibly apply transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
		KALDI_ASSERT(feats_transf.NumRows()==post.size()); 
//		KALDI_ASSERT(feats_transf.NumRows()==soft_post.NumRows()); 
        // add to cache
		feature_randomizer.AddData(feats_transf);
		targets_randomizer.AddData(post);
//		soft_randomizer.AddData(CuMatrix<BaseFloat>(soft_post));
        num_done++;
        
        // measure the time needed to get next feature file 
        Timer t_features;
        time_next += t_features.Elapsed();
        // report the speed
        if (num_done % 1000 == 0) {
          time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }

      }
      // randomize
      if (!crossvalidate && randomize) {
		const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
		feature_randomizer.Randomize(mask);
		targets_randomizer.Randomize(mask);
//		soft_randomizer.Randomize(mask);
      }
	  for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
										  targets_randomizer.Next()){
//										  soft_randomizer.Next()){
		nnet_in = feature_randomizer.Value();
		targets  = targets_randomizer.Value();
        nnet.Propagate(nnet_in, &nnet_out);
/*		soft = soft_randomizer.Value();
		PosteriorToMatrix(targets,post_dim,&post_mat);
		target_mat.Resize(rnd_opts.minibatch_size,post_dim*2);
		target_mat.ColRange(0,post_dim).CopyFromMat(post_mat);
		target_mat.ColRange(post_dim,post_dim).CopyFromMat(soft);*/
		if (objective_function == "xent") {
		xent.Eval(frm_weights,nnet_out,targets,&obj_diff);
		}else if (0 == objective_function.compare(0, 9, "multitask")) {
		multitask.Eval(frm_weights, nnet_out, targets, &obj_diff);
		}else {
          KALDI_ERR << "Unknown objective function code : " << objective_function;
        }

        if (!crossvalidate) {
          //nnet.LhnAdaptation(obj_diff, NULL);
          nnet.LhnAdaptation(obj_diff, NULL, mean, pre);
        }
        // monitor the training
        if (kaldi::g_kaldi_verbose_level >= 3) {
          if ((total_frames/100000) != ((total_frames+nnet_in.NumRows())/100000)) { // print every 100k frames
            if (!crossvalidate) {
              KALDI_VLOG(3) << nnet.InfoGradient();
            } else {
              KALDI_VLOG(3) << nnet.InfoPropagate();
            }
          }
        }
        total_frames += nnet_in.NumRows();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }
    
    KALDI_LOG << (crossvalidate?"CROSSVALIDATE":"TRAINING") << " FINISHED " 
              << time.Elapsed()/60 << "min, fps" << total_frames/time.Elapsed()
              << ", feature wait " << time_next << "s"; 

    KALDI_LOG << "Done " << num_done << " files, " << num_no_alignment
              << " with no alignments, " << num_other_error
              << " with other errors.";

	if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (0 == objective_function.compare(0, 9, "multitask")) {
      KALDI_LOG << multitask.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif


    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
