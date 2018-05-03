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
$ python train.py

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
WARNING:tensorflow:Using temporary folder as model directory: /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpe8pgnhv6
INFO:tensorflow:Using config: {'_model_dir': '/var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpe8pgnhv6', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x1119eee48>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
INFO:tensorflow:Calling model_fn.
==> Memory Episode 0
==> Memory Episode 1
==> Memory Episode 0
==> Memory Episode 1
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2018-05-04 00:01:34.476201: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Saving checkpoints for 1 into /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpe8pgnhv6/model.ckpt.
INFO:tensorflow:loss = 3.7616796, step = 1
INFO:tensorflow:global_step/sec: 3.03687
INFO:tensorflow:loss = 0.62995934, step = 101 (32.929 sec)
INFO:tensorflow:global_step/sec: 2.99219
INFO:tensorflow:loss = 0.49318463, step = 201 (33.420 sec)
INFO:tensorflow:global_step/sec: 3.06478
INFO:tensorflow:loss = 0.28764096, step = 301 (32.629 sec)
INFO:tensorflow:global_step/sec: 3.15028
INFO:tensorflow:loss = 0.17653497, step = 401 (31.743 sec)
INFO:tensorflow:global_step/sec: 3.00002
INFO:tensorflow:loss = 0.11094085, step = 501 (33.333 sec)
INFO:tensorflow:global_step/sec: 3.09383
INFO:tensorflow:loss = 0.15984775, step = 601 (32.323 sec)
INFO:tensorflow:global_step/sec: 3.10802
INFO:tensorflow:loss = 0.17948946, step = 701 (32.175 sec)
INFO:tensorflow:global_step/sec: 3.07913
INFO:tensorflow:loss = 0.11442192, step = 801 (32.477 sec)
INFO:tensorflow:global_step/sec: 3.09611
INFO:tensorflow:loss = 0.118585475, step = 901 (32.299 sec)
INFO:tensorflow:global_step/sec: 3.05136
INFO:tensorflow:loss = 0.17584072, step = 1001 (32.772 sec)
INFO:tensorflow:global_step/sec: 3.05374
INFO:tensorflow:loss = 0.12109965, step = 1101 (32.747 sec)
INFO:tensorflow:global_step/sec: 3.0906
INFO:tensorflow:loss = 0.13888617, step = 1201 (32.356 sec)
INFO:tensorflow:global_step/sec: 3.02957
INFO:tensorflow:loss = 0.09216529, step = 1301 (33.008 sec)
INFO:tensorflow:global_step/sec: 3.02622
INFO:tensorflow:loss = 0.048903227, step = 1401 (33.045 sec)
INFO:tensorflow:global_step/sec: 3.01416
INFO:tensorflow:loss = 0.018086608, step = 1501 (33.177 sec)
INFO:tensorflow:Saving checkpoints for 1563 into /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpe8pgnhv6/model.ckpt.
INFO:tensorflow:Loss for final step: 0.006601455.
INFO:tensorflow:Calling model_fn.
==> Memory Episode 0
==> Memory Episode 1
==> Memory Episode 0
==> Memory Episode 1
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from /var/folders/sx/fv0r97j96fz8njp14dt5g7940000gn/T/tmpe8pgnhv6/model.ckpt-1563
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
Testing Accuracy: 0.992

```
