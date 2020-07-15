


export BERT_BASE_DIR=/root/models/english/xlnet/tensorflow/xlnet_cased_L-24_H-1024_A-16


transformers-cli convert --model_type xlnet \
  --tf_checkpoint $BERT_BASE_DIR/xlnet_model.ckpt \
  --config $BERT_BASE_DIR/xlnet_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/