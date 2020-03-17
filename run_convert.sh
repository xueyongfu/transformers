


export BERT_BASE_DIR=/home/xyf/models/chinese/bert/tensorflow/multi_cased_L-12_H-768_A-12


transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin