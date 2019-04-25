// nnetbin/nnet-train-frmshuff.cc

// targeted attack
// atn aims to lead enh to output feats 
// alternatively we could use atn to lead enh to output noise (1-enh)

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  nnet-train-frmshuff [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-frmshuff scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts, atn_opts;
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
    std::string targets_transform;
    po.Register("targets-transform", &targets_transform, "Targets transform in Nnet format");
    std::string objective_function = "mse";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");
    BaseFloat loss_weight = 0.1,
              atn_lr = 1e-5;
    po.Register("loss-weight", &loss_weight, "weight of adversarial task");
    po.Register("atn-lr", &atn_lr, "learn rate of ATN");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
   
    bool freeze_enh = false, freeze_atn = false;
    po.Register("freeze_enh", &freeze_enh, "If true, do not modify top network");
    po.Register("freeze_atn", &freeze_atn, "If true, do not modify bot network");
    int32 repeat_atn = 1;
    po.Register("repeat-atn", &repeat_atn, "Train atn more per batch.");
 
    po.Read(argc, argv);

    if (po.NumArgs() != 6-(crossvalidate?2:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3),
      atn_filename = po.GetArg(4);
        
    std::string target_model_filename, target_atn_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(5);
      target_atn_filename = po.GetArg(6);
    }

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }
    Nnet nnet_transt;
    if (targets_transform != "") {
      nnet_transt.Read(targets_transform);
    }

    Nnet nnet,atn;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    atn_opts.learn_rate = atn_lr;
    atn_opts.momentum = trn_opts.momentum;
    atn.Read(atn_filename);
    atn.SetTrainOptions(atn_opts);
    KALDI_ASSERT(atn.InputDim() == atn.OutputDim());
    KALDI_ASSERT(atn.OutputDim() == nnet.InputDim());

    if (crossvalidate) {
      nnet_transf.SetDropoutRate(0.0);
      nnet.SetDropoutRate(0.0);
      atn.SetDropoutRate(0.0);
    }

    int32 dim_in = nnet.InputDim();
    int32 dim_out = nnet.OutputDim();
    int32 splice = (int)(dim_in/dim_out);
    int32 half_splice = (int)((splice-1)/2);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Mse mse(loss_opts);
    Mse mse3(loss_opts),mse2(loss_opts),mse4(loss_opts);
    MultiTaskLoss multitask(loss_opts); 
    if (0 == objective_function.compare(0, 9, "multitask")) {
      // objective_function contains something like :
      // 'multitask,xent,2456,1.0,mse,440,0.001'
      //
      // the meaning is following:
      // 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
      multitask.InitFromString(objective_function);
    }
    CuMatrix<BaseFloat> feats_transf, nnet_out, tgt_transf, obj_diff,in_diff;
    CuMatrix<BaseFloat> atn_out,ae_diff,adv_diff;
    CuMatrix<BaseFloat> nnet_in_center(rnd_opts.minibatch_size,nnet.OutputDim());
    int32 batch_counter = 0;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (!feature_reader.Done() && !targets_reader.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !feature_reader.Done() && !targets_reader.Done(); feature_reader.Next(), targets_reader.Next()) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // check that we have targets
        if (targets_reader.Key() != utt) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          break;
        }
        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Matrix<BaseFloat> targets = targets_reader.Value();
        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }
        // correct small length mismatch ... or drop sentence
        {
          // add lengths to vector
          std::vector<int32> lenght;
          lenght.push_back(mat.NumRows());
          lenght.push_back(targets.NumRows());
          lenght.push_back(weights.Dim());
          // find min, max
          int32 min = *std::min_element(lenght.begin(),lenght.end());
          int32 max = *std::max_element(lenght.begin(),lenght.end());
          // fix or drop ?
          if (max - min < length_tolerance) {
            if(mat.NumRows() != min) mat.Resize(min, mat.NumCols(), kCopyData);
            if(targets.NumRows() != min) targets.Resize(min, targets.NumCols(), kCopyData);
            if(weights.Dim() != min) weights.Resize(min, kCopyData);
          } else {
            KALDI_WARN << utt << ", length mismatch of targets " << targets.NumRows()
                       << " and features " << mat.NumRows();
            num_other_error++;
            continue;
          }
        }
        // apply optional feature transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
        nnet_transt.Feedforward(CuMatrix<BaseFloat>(targets), &tgt_transf);

        // pass data to randomizers
        KALDI_ASSERT(feats_transf.NumRows() == tgt_transf.NumRows());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(tgt_transf);
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
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
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat>& nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass
        if (!freeze_enh){
            nnet.Propagate(nnet_in, &nnet_out);
            if (objective_function == "mse") {
              mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
            } else if (0 == objective_function.compare(0, 9, "multitask")) {
              multitask.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
            } else {
              KALDI_ERR << "Unknown objective function code : " << objective_function;
            }
            // backward pass
            if (!crossvalidate) {
              nnet.Backpropagate(obj_diff, NULL);
            }
        }
        if (!freeze_atn){
        //ATN Training
            for (int32 jj = 0; jj < repeat_atn; jj++){
                atn.Propagate(nnet_in, &atn_out);
                mse2.Eval(frm_weights,atn_out,nnet_in, &ae_diff);
                nnet_in_center.CopyFromMat(nnet_in.ColRange(dim_out*half_splice,dim_out));//adv target

                nnet.Propagate(atn_out, &nnet_out);
                mse3.Eval(frm_weights,nnet_out,nnet_in_center, &adv_diff);
                if (!crossvalidate){
                  nnet.Feedbackward(adv_diff, &in_diff);
                  ae_diff.AddMat(loss_weight,in_diff);
                  atn.Backpropagate(ae_diff, NULL);
                }
                if(mse3.BatchLoss()<mse.BatchLoss())
                    break;
            }
        }
        if(!freeze_enh){
            //now retrain enh
            atn.Propagate(nnet_in, &atn_out);
            nnet.Propagate(atn_out, &nnet_out);
            mse4.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
            if (! crossvalidate)
              nnet.Backpropagate(obj_diff, NULL);
        }
        batch_counter++;
        KALDI_LOG << "Batch " << batch_counter << " : " << mse.BatchLoss() << ", "
                << mse2.BatchLoss() << ", " << mse3.BatchLoss() << ", "
                << mse4.BatchLoss() << ".";
        total_frames += nnet_in.NumRows();
      }
    }
    if (!crossvalidate) {
      if(!freeze_enh)
          nnet.Write(target_model_filename, binary);
      if(!freeze_atn)
          atn.Write(target_atn_filename,binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    if (objective_function == "mse") {
      KALDI_LOG << "mse1: " << mse.Report();
      KALDI_LOG << "mse2: " << mse2.Report();
      KALDI_LOG << "mse3: " << mse3.Report();
      KALDI_LOG << "mse4: " << mse4.Report();
      if(!freeze_enh){
        KALDI_LOG << "ATN training loss (wMse): " << mse.AvgLoss()+loss_weight*mse4.AvgLoss() << " .";
      }
      if(!freeze_atn){
        KALDI_LOG << "ATN training loss (wMse): " << mse2.AvgLoss()+loss_weight*mse3.AvgLoss() << " .";
      }
      
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
