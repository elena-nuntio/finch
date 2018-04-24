---
Implementing the idea of ["Attention is All you Need"](https://arxiv.org/abs/1706.03762)

---

<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/transformer.png" width="300">

<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/multihead_attn.png" width='700'>

Some functions are adapted from [Kyubyong's](https://github.com/Kyubyong/transformer) work, thanks for him!

* Based on that, we have:
    * implemented the model under the architecture of ```tf.estimator.Estimator``` API

    * added an option to share the weights between encoder embedding and decoder embedding

    * added an option to share the weights between decoder embedding and output projection

    * added the learning rate variation according to the formula in paper, and also expotential decay

    * added more activation choices (leaky relu / elu) for easier gradient propagation

    * fixed masking mistake discovered [here](https://github.com/Kyubyong/transformer/issues/3)

    * used ```tf.while_loop``` to perform autoregressive decoding on graph, instead of ```feed_dict```

* Small Task 1: learn sorting characters

    ```  python train_letters.py --tied_embedding --label_smoothing ```
        
    ```
   INFO:tensorflow:Loss for final step: 0.69530267.
   INFO:tensorflow:Calling model_fn.
   INFO:tensorflow:Done calling model_fn.
   INFO:tensorflow:Graph was finalized.
   INFO:tensorflow:Restoring parameters from ./saved/model.ckpt-5000
   INFO:tensorflow:Running local_init_op.
   INFO:tensorflow:Done running local_init_op.
   apple -> aelpp
   common -> cmmnoo
   zhedong -> deghnoz
    ```

* Small Task 2: learn chinese dialog

    ``` python train_dialog.py```
    
    ```
    INFO:tensorflow:Loss for final step: 4.581911.
    INFO:tensorflow:Calling model_fn.
    INFO:tensorflow:Done calling model_fn.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./saved/model.ckpt-7092
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    你是谁 -> 我是小通
    你喜欢我吗 -> 我喜欢你
    给我唱一首歌 -> =。=========
    我帅吗 -> 你是我的
    ```

<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/transform20fps.gif" height='400'>
