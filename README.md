# Pre-Training of German T5 models

This repository gives an overview of how-to pre-train T5 models for German from scratch.

# Preparing GCP bucket

The T5 library expects the training dataset (created via TensorFlow Datasets) on a GCP bucket. First, the `gcloud` utils needs to be installed. This can be done in a Docker container via:

```bash
$ curl -sSL https://sdk.cloud.google.com | bash
$ gcloud auth login
```

Then you need to authorize via your Google account and copy-paste the corresponding token. In the next you need to create a GCP bucket in the same zone as your TPU. Please give "Storage Administrator" permissions to your `service*` account in the GCP bucket web console. After creating the GCP bucket, just create a folder `datasets`, where all TensorFlow Datasets can be uploaded to:

```bash
$ gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -r gc4_dataset gs://german-t5-gc4/datasets
```

In this example we upload our previously created `gc4_dataset`, that has the following folder structure:

```bash
$ tree gc4_dataset/
gc4_dataset/
├── __init__.py
├── __pycache__
│   └── gc4_dataset.cpython-38.pyc
├── checksums.tsv
├── dummy_data
│   └── TODO-add_fake_data_in_this_directory.txt
├── gc4_dataset.py
└── gc4_dataset_test.py
├── 1.0.0
│   └── gc4_dataset-train.tfrecord-00000-of-01024
│   └── ...
```

The `1.0.0` folder contains all created TFRecords. Uploading into your GCP bucket can be done with:

```bash
gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp -r gc4_dataset gs://german-t5-gc4/datasets
```

In the last step you also need to upload your previously created vocab (`spiece.model`) via:

```bash
$ gsutil cp spiece.model gs://german-t5-gc4
```

so it will be stored at root level of the GCP bucket.


# Preparing VM

In the next step, a VM needs to be created to coordinate the pre-training process. We are using a `n1-standard-2` instance and a customized boot disk size of 50GB. Please notice that the default boot disk size is 10GB, which is not enough for all Python dependencies. We are using TensorFlow 2.8 in our experiments:

```bash
$ gcloud compute instances create t5 --zone=us-central1-a \
  --machine-type=n1-standard-2 --image-project=ml-images \
  --image-family=tf-2-8-0 --scopes=cloud-platform \
  --boot-disk-size 50GB
```

The VM should be in the same zone as GCP bucket and TPU.

After VM creation, we can SSH into it:

```bash
$ gcloud compute ssh t5 --zone us-central1-a
```

Now we immediately start a `tmux` session and run all commands in this session. If the connection to the VM got lost, you can re-sume the session with `tmux attach` after next login.

```bash
$ tmux
```

## Dependencies

We just need to clone the T5 repository:

```bash
$ git clone https://github.com/google-research/text-to-text-transfer-transformer.git
$ cd text-to-text-transfer-transformer
$ pip3 install -e .
$ export PATH=$PATH:$HOME/.local/bin
```

This will install all necessary dependencies. To make sure that everything is working, just run:

```bash
$ t5_mesh_transformer --helpfull
```

# Custom Task

To pre-train a model from scratch on our own corpus, we need to slightly extend the T5 library. To do so, we just our corpus to the internal task registry. This can be done in the `t5/data/tasks.py` script of the `text-to-text-transfer-transformer` folder. The following lines need to be prepended:

```python
SPM_VOCAB = "gs://german-t5-gc4/spiece.model"

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(SPM_VOCAB, extra_ids=100), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=seqio.SentencePieceVocabulary(SPM_VOCAB, extra_ids=100), add_eos=True)
}

# Custom
TaskRegistry.add(
    "gc4_corpus",
    source=seqio.TfdsDataSource(tfds_name="gc4_dataset:1.0.0"),
    preprocessors=[
        functools.partial(
            preprocessors.rekey, key_map={
                "inputs": None,
                "targets": "text"
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[],
    )
```

# Model configuration

The model configuration is stored in a so called *GIN* configuration file. For training a 32EL Small model (as proposed in the [Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers](https://arxiv.org/abs/2109.10686) paper, we use the following modified *GIN* configuration:

```python
import mesh_tensorflow.optimize
import mesh_tensorflow.transformer.dataset
import mesh_tensorflow.transformer.learning_rate_schedules
import mesh_tensorflow.transformer.t2t_vocabulary
import mesh_tensorflow.transformer.transformer
import mesh_tensorflow.transformer.transformer_layers
import mesh_tensorflow.transformer.utils
import t5.models.mesh_transformer

# Macros:
# ==============================================================================
d_ff = 2048
d_kv = 64
d_model = 512
dropout_rate = 0.0
inputs_length = 512
mean_noise_span_length = 3.0
MIXTURE_NAME = 'gc4_corpus'
noise_density = 0.15
num_heads = 8
num_layers = 6

# Parameters for adafactor_decay_rate_pow:
# ==============================================================================
adafactor_decay_rate_pow.offset = 0

# Parameters for AdafactorOptimizer:
# ==============================================================================
AdafactorOptimizer.beta1 = 0.0
AdafactorOptimizer.clipping_threshold = 1.0
AdafactorOptimizer.decay_rate = None
AdafactorOptimizer.epsilon1 = 1e-30
AdafactorOptimizer.epsilon2 = 0.001
AdafactorOptimizer.factored = True
AdafactorOptimizer.min_dim_size_to_factor = 128
AdafactorOptimizer.multiply_by_parameter_scale = True

# Parameters for Bitransformer:
# ==============================================================================
Bitransformer.shared_embedding = True

# Parameters for denoise:
# ==============================================================================
denoise.inputs_fn = @preprocessors.noise_span_to_unique_sentinel
denoise.noise_density = %noise_density
denoise.noise_mask_fn = @preprocessors.random_spans_noise_mask
denoise.targets_fn = @preprocessors.nonnoise_span_to_unique_sentinel

# Parameters for decoder/DenseReluDense:
# ==============================================================================
decoder/DenseReluDense.activation = 'relu'
decoder/DenseReluDense.dropout_rate = %dropout_rate
decoder/DenseReluDense.hidden_size = %d_ff
decoder/DenseReluDense.use_bias = False

# Parameters for encoder/DenseReluDense:
# ==============================================================================
encoder/DenseReluDense.activation = 'relu'
encoder/DenseReluDense.dropout_rate = %dropout_rate
encoder/DenseReluDense.hidden_size = %d_ff
encoder/DenseReluDense.use_bias = False

# Parameters for enc_dec_attention:
# ==============================================================================
# None.

# Parameters for enc_dec_attention_bias:
# ==============================================================================
# None.

# Parameters for decoder/EncDecAttention:
# ==============================================================================
decoder/EncDecAttention.relative_attention_type = None

# Parameters for get_variable_dtype:
# ==============================================================================
get_variable_dtype.activation_dtype = 'bfloat16'

# Parameters for get_vocab_embedding_cls:
# ==============================================================================
# None.

# Parameters for get_vocabulary:
# ==============================================================================
get_vocabulary.mixture_or_task_name = %MIXTURE_NAME

# Parameters for decoder/LayerStack:
# ==============================================================================
decoder/LayerStack.dropout_rate = None
decoder/LayerStack.norm_epsilon = None
decoder/LayerStack.recompute_grads = False
decoder/LayerStack.sublayers_final = \
[@transformer.sublayer_rms_norm, @transformer.sublayer_dropout]
decoder/LayerStack.sublayers_initial = [@transformer.sublayer_dropout]
decoder/LayerStack.sublayers_per_layer = \
[@transformer.sublayer_rms_norm,
@transformer.sublayer_call_layer,
@transformer.sublayer_dropout,
@transformer.sublayer_residual]

# Parameters for encoder/LayerStack:
# ==============================================================================
encoder/LayerStack.dropout_rate = None
encoder/LayerStack.norm_epsilon = None
encoder/LayerStack.recompute_grads = False
encoder/LayerStack.sublayers_final = \
[@transformer.sublayer_rms_norm, @transformer.sublayer_dropout]
encoder/LayerStack.sublayers_initial = [@transformer.sublayer_dropout]
encoder/LayerStack.sublayers_per_layer = \
[@transformer.sublayer_rms_norm,
@transformer.sublayer_call_layer,
@transformer.sublayer_dropout,
@transformer.sublayer_residual]

# Parameters for learning_rate_schedule_noam:
# ==============================================================================
learning_rate_schedule_noam.linear_decay_fraction = 0.0
learning_rate_schedule_noam.multiplier = 1.0
learning_rate_schedule_noam.offset = 0
learning_rate_schedule_noam.warmup_steps = 10000

# Parameters for make_bitransformer:
# ==============================================================================
make_bitransformer.decoder_name = 'decoder'
make_bitransformer.encoder_name = 'encoder'

# Parameters for decoder/make_layer_stack:
# ==============================================================================
decoder/make_layer_stack.block_scope = True
decoder/make_layer_stack.layers = \
[@mesh_tensorflow.transformer.transformer_layers.SelfAttention,
@mesh_tensorflow.transformer.transformer_layers.EncDecAttention,
@mesh_tensorflow.transformer.transformer_layers.DenseReluDense]
decoder/make_layer_stack.num_layers = %num_layers

# Parameters for encoder/make_layer_stack:
# ==============================================================================
encoder/make_layer_stack.block_scope = True
encoder/make_layer_stack.layers = \
[@mesh_tensorflow.transformer.transformer_layers.SelfAttention,
@mesh_tensorflow.transformer.transformer_layers.DenseReluDense]
encoder/make_layer_stack.num_layers = 32

# Parameters for mesh_train_dataset_fn:
# ==============================================================================
mesh_train_dataset_fn.mixture_or_task_name = %MIXTURE_NAME
mesh_train_dataset_fn.pack = True
mesh_train_dataset_fn.seed = None
mesh_train_dataset_fn.shuffle = True
mesh_train_dataset_fn.use_cached = False

# Parameters for noise_span_to_unique_sentinel:
# ==============================================================================
# None.

# Parameters for nonnoise_span_to_unique_sentinel:
# ==============================================================================
# None.

# Parameters for pack_dataset:
# ==============================================================================
pack_dataset.use_custom_ops = False

# Parameters for pack_or_pad:
# ==============================================================================
# None.

# Parameters for random_spans_helper:
# ==============================================================================
random_spans_helper.extra_tokens_per_span_inputs = 1
random_spans_helper.extra_tokens_per_span_targets = 1
random_spans_helper.inputs_length = %inputs_length
random_spans_helper.mean_noise_span_length = %mean_noise_span_length
random_spans_helper.noise_density = %noise_density
random_spans_helper.verbose = False

# Parameters for random_spans_noise_mask:
# ==============================================================================
random_spans_noise_mask.mean_noise_span_length = %mean_noise_span_length

# Parameters for random_spans_tokens_length:
# ==============================================================================
# None.

# Parameters for reduce_concat_tokens:
# ==============================================================================
reduce_concat_tokens.batch_size = 128
reduce_concat_tokens.feature_key = 'targets'

# Parameters for rewrite_stack_variables:
# ==============================================================================
rewrite_stack_variables.max_combined_variable_size = 536870912

# Parameters for run:
# ==============================================================================
run.autostack = True
run.batch_size = ('tokens_per_batch', 65536)
run.checkpoint_input_pipeline = False
run.dataset_split = 'train'
run.ensemble_inputs = None
run.eval_checkpoint_step = None
run.eval_dataset_fn = None
run.eval_summary_dir = None
run.export_checkpoint_step = None
run.export_path = ''
run.init_checkpoint = None
run.iterations_per_loop = 100
run.keep_checkpoint_max = None
run.layout_rules = \
'ensemble:ensemble,batch:batch,d_ff:model,heads:model,vocab:model,experts:batch'
run.learning_rate_schedule = @learning_rate_schedules.learning_rate_schedule_noam
run.mesh_devices = None
run.mesh_shape = @mesh_tensorflow.transformer.utils.tpu_mesh_shape()
run.mode = 'train'
run.model_type = 'bitransformer'
run.optimizer = @optimize.AdafactorOptimizer
run.output_eval_examples = True
run.perplexity_eval_steps = 100
run.predict_fn = None
run.save_checkpoints_steps = 50000
run.seen_data_init_step = 0
run.sequence_length = {'inputs': 512, 'targets': 128}
run.skip_seen_data = False
run.total_run_steps = None
run.train_dataset_fn = @t5.models.mesh_transformer.mesh_train_dataset_fn
run.train_steps = 524288
run.variable_filter = None

# Parameters for select_random_chunk:
# ==============================================================================
select_random_chunk.additional_feature_keys = None
select_random_chunk.additional_passthrough_keys = None
select_random_chunk.feature_key = 'targets'
select_random_chunk.max_length = 65536
select_random_chunk.uniform_random_start = False

# Parameters for decoder/SelfAttention:
# ==============================================================================
decoder/SelfAttention.attention_func = None
decoder/SelfAttention.attention_kwargs = None
decoder/SelfAttention.combine_dims = True
decoder/SelfAttention.dropout_rate = %dropout_rate
decoder/SelfAttention.fold_scaling_into_initializer = True
decoder/SelfAttention.keep_query_heads_dims = False
decoder/SelfAttention.key_value_size = %d_kv
decoder/SelfAttention.num_heads = %num_heads
decoder/SelfAttention.num_memory_heads = 0
decoder/SelfAttention.relative_attention_num_buckets = 32
decoder/SelfAttention.relative_attention_type = 'bias_shared'
decoder/SelfAttention.shared_kv = False

# Parameters for encoder/SelfAttention:
# ==============================================================================
encoder/SelfAttention.attention_func = None
encoder/SelfAttention.attention_kwargs = None
encoder/SelfAttention.combine_dims = True
encoder/SelfAttention.dropout_rate = %dropout_rate
encoder/SelfAttention.fold_scaling_into_initializer = True
encoder/SelfAttention.keep_query_heads_dims = False
encoder/SelfAttention.key_value_size = %d_kv
encoder/SelfAttention.num_heads = %num_heads
encoder/SelfAttention.num_memory_heads = 0
encoder/SelfAttention.relative_attention_num_buckets = 32
encoder/SelfAttention.relative_attention_type = 'bias_shared'
encoder/SelfAttention.shared_kv = False

# Parameters for serialize_num_microbatches:
# ==============================================================================
serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192

# Parameters for SimdMeshImpl:
# ==============================================================================
SimdMeshImpl.allreduce_in_bfloat16_max_group_size = 8

# Parameters for split_tokens:
# ==============================================================================
split_tokens.additional_feature_keys = None
split_tokens.feature_key = 'targets'
split_tokens.max_tokens_per_segment = @preprocessors.random_spans_tokens_length()
split_tokens.min_tokens_per_segment = None
split_tokens.passthrough_feature_keys = None

# Parameters for sublayer_call_layer:
# ==============================================================================
# None.

# Parameters for sublayer_dropout:
# ==============================================================================
sublayer_dropout.dropout_rate = %dropout_rate

# Parameters for sublayer_mask_padding:
# ==============================================================================
# None.

# Parameters for sublayer_residual:
# ==============================================================================
# None.

# Parameters for sublayer_rms_norm:
# ==============================================================================
sublayer_rms_norm.epsilon = 1e-06
sublayer_rms_norm.name = 'rms_norm'

# Parameters for tpu_estimator_model_fn:
# ==============================================================================
tpu_estimator_model_fn.hierarchical_tiling_spec = None
tpu_estimator_model_fn.init_variable_filter = ''
tpu_estimator_model_fn.model_info_file = ''
tpu_estimator_model_fn.outer_batch_size = 1
tpu_estimator_model_fn.tpu_summaries = False

# Parameters for tpu_mesh_shape:
# ==============================================================================
tpu_mesh_shape.ensemble_parallelism = None
tpu_mesh_shape.model_parallelism = 1
tpu_mesh_shape.tpu_topology = '4x4'

# Parameters for unit_scaling_convention:
# ==============================================================================
unit_scaling_convention.value = False

# Parameters for decoder/Unitransformer:
# ==============================================================================
decoder/Unitransformer.d_model = %d_model
decoder/Unitransformer.ensemble = None
decoder/Unitransformer.input_full_attention = False
decoder/Unitransformer.label_smoothing = 0.0
decoder/Unitransformer.loss_denominator = None
decoder/Unitransformer.loss_fn = None
decoder/Unitransformer.loss_on_targets_only = False
decoder/Unitransformer.max_length = 512
decoder/Unitransformer.positional_embedding = False
decoder/Unitransformer.shared_embedding_and_softmax_weights = True
decoder/Unitransformer.sinusoid_positional_embedding = False
decoder/Unitransformer.token_dropout_rate = 0.0
decoder/Unitransformer.vocab_divisor = 128
decoder/Unitransformer.z_loss = 0.0001

# Parameters for encoder/Unitransformer:
# ==============================================================================
encoder/Unitransformer.d_model = %d_model
encoder/Unitransformer.ensemble = None
encoder/Unitransformer.input_full_attention = False
encoder/Unitransformer.label_smoothing = 0.0
encoder/Unitransformer.loss_denominator = None
encoder/Unitransformer.loss_fn = None
encoder/Unitransformer.loss_on_targets_only = False
encoder/Unitransformer.max_length = 512
encoder/Unitransformer.positional_embedding = False
encoder/Unitransformer.shared_embedding_and_softmax_weights = True
encoder/Unitransformer.sinusoid_positional_embedding = False
encoder/Unitransformer.token_dropout_rate = 0.0
encoder/Unitransformer.vocab_divisor = 128
encoder/Unitransformer.z_loss = 0.0001

# Parameters for unsupervised:
# ==============================================================================
unsupervised.preprocessors = \
[@preprocessors.select_random_chunk,
@preprocessors.reduce_concat_tokens,
@preprocessors.split_tokens,
@preprocessors.denoise]

# Parameters for VarianceScalingInitializer:
# ==============================================================================
VarianceScalingInitializer.distribution = 'normal'
VarianceScalingInitializer.mode = 'fan_in'
VarianceScalingInitializer.scale = 1.0

# Parameters for VocabEmbedding:
# ==============================================================================
VocabEmbedding.scale_variable_like_classifier_weights = False
```

Then save the configuration under `$HOME/operative_config.gin`.

We performed minor changes compared to the official *GIN* configuration:

* `MIXTURE_NAME` is set to our own corpus (as previously added to the T5 Task Registry), named `gc4_corpus`.
* `mesh_train_dataset_fn.use_cached` is set to `False`, because we use our own dataset on GCP bucket.
* `pack_dataset.use_custom_ops` is set to `False`, because otherwise an own-compiled Tensor2Tensor Ops package needs to be installed. We currently have no idea of how to do this, because it is undocumented, see this [issue](https://github.com/tensorflow/tensor2tensor/issues/1846) for more information.
* `run.save_checkpoints_steps` is set to 50,000 steps.

# Create TPU

In the next, we need to create a TPU for pre-training. Please adjust your `zone` (mentioned in the email from TRC) and your TPU type (v3-8 is used here):

```bash
$ gcloud compute tpus create t5 --zone=us-central1-a \
  --accelerator-type=v3-8 --network=default \
  --range=192.168.9.0/29  --version=2.8.0
```

# Pre-Training

The final pre-training can be started with:

```bash
$ t5_mesh_transformer --tpu=t5 --gcp_project=project-name \
  --tpu_zone=us-central1-a --model_dir=gs://german-t5-gc4/t5-small-german \
  --gin_file=$HOME/operative_config.gin \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
  --t5_tfds_data_dir=gs://german-t5-gc4/datasets

```

Please make sure that you've adjusted the correct `gcp_project` and `tpu_zone`.
