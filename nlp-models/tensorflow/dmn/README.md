<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/dmn-details.png">

---

Implementing the idea of ["
Dynamic Memory Networks for Visual and Textual Question Answering"](https://arxiv.org/abs/1603.01417)

---

* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:
    * We have used ```tf.estimator.Estimator``` API to package the model
    * We have used ```tf.map_fn``` to replace the Python for loop, which makes the model truly dynamic
    * We have added a decoder in the answer module for "talking"
    * We have reproduced ```AttentionGRUCell``` from new official ```GRUCell```

---

```
python train.py
```
```
{
    "n_epochs": 10,
    "batch_size": 64,
    "embed_dim": 80,
    "hidden_size": 80,
    "dropout_rate": 0.1,
    "n_hops": 2,
    "clip_norm": 5.0
}
INFO:tensorflow:Using default config.
WARNING:tensorflow:Using temporary folder as model directory: /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpn5tsxe2f
INFO:tensorflow:Using config: {'_model_dir': '/var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpn5tsxe2f', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x11f19bdd8>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
==> Memory Episode 0
==> Memory Episode 1
==> Memory Episode 0
==> Memory Episode 1
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-05-03 22:32:31.218865: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpn5tsxe2f/model.ckpt.
INFO:tensorflow:loss = 3.7668605, step = 1
INFO:tensorflow:global_step/sec: 3.02908
INFO:tensorflow:loss = 0.6363834, step = 101 (33.014 sec)
INFO:tensorflow:global_step/sec: 3.06571
INFO:tensorflow:loss = 0.5419112, step = 201 (32.618 sec)
INFO:tensorflow:global_step/sec: 3.12723
INFO:tensorflow:loss = 0.3097182, step = 301 (31.977 sec)
INFO:tensorflow:global_step/sec: 3.03866
INFO:tensorflow:loss = 0.24448162, step = 401 (32.909 sec)
INFO:tensorflow:global_step/sec: 3.08223
INFO:tensorflow:loss = 0.19115436, step = 501 (32.444 sec)
INFO:tensorflow:global_step/sec: 3.11966
INFO:tensorflow:loss = 0.21831298, step = 601 (32.055 sec)
INFO:tensorflow:global_step/sec: 3.12905
INFO:tensorflow:loss = 0.1209056, step = 701 (31.959 sec)
INFO:tensorflow:global_step/sec: 3.09877
INFO:tensorflow:loss = 0.15908256, step = 801 (32.271 sec)
INFO:tensorflow:global_step/sec: 3.13336
INFO:tensorflow:loss = 0.052731514, step = 901 (31.915 sec)
INFO:tensorflow:global_step/sec: 3.10586
INFO:tensorflow:loss = 0.03756873, step = 1001 (32.197 sec)
INFO:tensorflow:global_step/sec: 3.10244
INFO:tensorflow:loss = 0.0525392, step = 1101 (32.233 sec)
INFO:tensorflow:global_step/sec: 3.0645
INFO:tensorflow:loss = 0.008978036, step = 1201 (32.631 sec)
INFO:tensorflow:global_step/sec: 3.14752
INFO:tensorflow:loss = 0.033921257, step = 1301 (31.771 sec)
INFO:tensorflow:global_step/sec: 3.11648
INFO:tensorflow:loss = 0.049431074, step = 1401 (32.088 sec)
INFO:tensorflow:global_step/sec: 3.13156
INFO:tensorflow:loss = 0.031650156, step = 1501 (31.933 sec)
INFO:tensorflow:Saving checkpoints for 1563 into /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpn5tsxe2f/model.ckpt.
INFO:tensorflow:Loss for final step: 0.007123784.
INFO:tensorflow:Calling model_fn.
==> Memory Episode 0
==> Memory Episode 1
==> Memory Episode 0
==> Memory Episode 1
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpn5tsxe2f/model.ckpt-1563
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
Testing Accuracy: 0.992
```
