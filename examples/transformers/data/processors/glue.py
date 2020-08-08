# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """

import logging
import os

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        # print(len(inputs["input_ids"]))
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, lines):
        """See base class."""
        return self._create_examples(lines, "dev")

    def get_labels(self):
        """See base class."""
        # return ['正面', '负面', '中性']
        return ['公安政法类_公安(出入境)_其他_其他', '公安政法类_公安(出入境)_护照办理_护照办理',
       '公安政法类_公安（治安、交通）_交通管理_交通事故', '公安政法类_公安（治安、交通）_交通管理_交通法规',
       '公安政法类_公安（治安、交通）_交通管理_交通设施', '公安政法类_公安（治安、交通）_交通管理_交通违章',
       '公安政法类_公安（治安、交通）_交通管理_排堵保畅', '公安政法类_公安（治安、交通）_交通管理_无主废弃车辆',
       '公安政法类_公安（治安、交通）_交通管理_车辆年检', '公安政法类_公安（治安、交通）_交通管理_过户上牌',
       '公安政法类_公安（治安、交通）_交通管理_限流限行', '公安政法类_公安（治安、交通）_交通管理_非法客运',
       '公安政法类_公安（治安、交通）_交通管理_驾驶员审验', '公安政法类_公安（治安、交通）_其他_其他',
       '公安政法类_公安（治安、交通）_户籍管理_外省市迁入', '公安政法类_公安（治安、交通）_户籍管理_市内户口迁移',
       '公安政法类_公安（治安、交通）_户籍管理_户籍政策咨询', '公安政法类_公安（治安、交通）_户籍管理_收养审批',
       '公安政法类_公安（治安、交通）_户籍管理_死亡登记', '公安政法类_公安（治安、交通）_户籍管理_身份证管理',
       '公安政法类_公安（治安、交通）_特种行业管理_娱乐场所管理', '公安政法类_公安（治安、交通）_特种行业管理_行政许可',
       '公安政法类_公安（治安、交通）_特种行业管理_黑网吧', '公安政法类_公安（治安、交通）_犬类管理_养犬登记',
       '公安政法类_公安（治安、交通）_犬类管理_无证犬类', '公安政法类_公安（治安、交通）_犬类管理_流浪犬收治',
       '公安政法类_公安（治安、交通）_社会噪声_外置扩音装置', '公安政法类_公安（治安、交通）_社会噪声_装修噪音',
       '公安政法类_公安（治安、交通）_社会噪声_违规鸣笛', '公安政法类_公安（治安、交通）_社会治安_公安信息查询',
       '公安政法类_公安（治安、交通）_社会治安_刑事案件', '公安政法类_公安（治安、交通）_社会治安_危险品管理',
       '公安政法类_公安（治安、交通）_社会治安_居住证', '公安政法类_公安（治安、交通）_社会治安_治安维护',
       '公安政法类_公安（治安、交通）_社会治安_流浪人员管理', '公安政法类_公安（治安、交通）_社会治安_游行集会',
       '公安政法类_公安（治安、交通）_社会治安_立案侦查', '公安政法类_公安（治安、交通）_社会治安_经济案件',
       '公安政法类_公安（治安、交通）_社会治安_网络安全', '公安政法类_公安（治安、交通）_社会治安_黄赌毒',
       '公安政法类_公安（消防）_其他_其他', '公安政法类_公安（消防）_审批许可_审批许可',
       '公安政法类_公安（消防）_消防管理_易燃易爆物', '公安政法类_公安（消防）_消防管理_消防安全咨询',
       '公安政法类_公安（消防）_消防管理_消防宣传', '公安政法类_公安（消防）_消防管理_消防隐患',
       '公安政法类_公安（消防）_消防管理_火灾报警', '公安政法类_公安（消防）_消防管理_烟花爆竹管理',
       '公安政法类_公安（消防）_消防设备维护_消防栓', '公安政法类_公安（消防）_消防设备维护_消防通道',
       '公安政法类_公安（消防）_消防设备维护_火灾报警系统', '公安政法类_公安（消防）_消防设备维护_灭火器',
       '公安政法类_司法行政_公证服务_其他公证', '公安政法类_司法行政_公证服务_无犯罪证明',
       '公安政法类_司法行政_公证服务_继承权公证', '公安政法类_司法行政_其他_其他',
       '公安政法类_司法行政_司法直属_监狱管理', '公安政法类_司法行政_司法直属_矫正办',
       '公安政法类_司法行政_法律服务_司法鉴定', '公安政法类_司法行政_法律服务_律师服务',
       '公安政法类_司法行政_法律服务_普法宣传', '公安政法类_司法行政_法律服务_法律援助',
       '公安政法类_司法行政_纠纷仲裁_人民调解', '公安政法类_司法行政_纠纷仲裁_劳动争议',
       '公安政法类_司法行政_纠纷仲裁_经济纠纷', '公安政法类_国家安全_其他_其他',
       '公安政法类_国家安全_国家安全_国内安全事务', '公安政法类_国家安全_国家安全_国家机密',
       '公安政法类_检察院系统_其他_其他', '公安政法类_法院系统_其他_其他', '公安政法类_法院系统_法院信息_法院地址电话',
       '公安政法类_法院系统_法院受理_诉讼立案', '公用事业类_供水_供水报修_停水水小', '公用事业类_供水_供水报修_其他报修',
       '公用事业类_供水_供水报修_水管维修', '公用事业类_供水_供水报修_水表失窃', '公用事业类_供水_供水报修_水表维修',
       '公用事业类_供水_供水服务_业务办理', '公用事业类_供水_供水服务_信息查询', '公用事业类_供水_供水服务_服务质量',
       '公用事业类_供水_供水计费_水费量高', '公用事业类_供水_供水计费_缴费信息', '公用事业类_供水_供水计费_账单抄表',
       '公用事业类_供水_其他_其他', '公用事业类_供水_自来水水质_水中杂质', '公用事业类_供水_自来水水质_自来水异味',
       '公用事业类_供水_自来水水质_颜色异变', '公用事业类_供水_违规用水_私接自来水管',
       '公用事业类_供水_违规用水_私移水表', '公用事业类_气象_其他_其他', '公用事业类_气象_气象预报_天气预报',
       '公用事业类_气象_气象预报_气象服务信息', '公用事业类_气象_气象预报_环境气象', '公用事业类_气象_气象预报_空气指数',
       '公用事业类_气象_气象预警_台风沙尘', '公用事业类_气象_气象预警_暴雨暴雪', '公用事业类_气象_气象预警_道路结冰预警',
       '公用事业类_气象_气象预警_雷电预警', '公用事业类_气象_气象预警_高温预警', '公用事业类_燃气_其他_其他',
       '公用事业类_燃气_燃气报修_燃气泄漏', '公用事业类_燃气_燃气报修_燃气火小', '公用事业类_燃气_燃气报修_燃气配件',
       '公用事业类_燃气_燃气报修_管道维修', '公用事业类_燃气_燃气服务_业务办理', '公用事业类_燃气_燃气服务_信息咨询',
       '公用事业类_燃气_燃气服务_安全检查', '公用事业类_燃气_燃气服务_服务质量', '公用事业类_燃气_燃气服务_燃气施工',
       '公用事业类_燃气_燃气计费_用量偏高', '公用事业类_燃气_燃气计费_缴费信息', '公用事业类_燃气_燃气计费_账单抄表',
       '公用事业类_电力_其他_其他', '公用事业类_电力_故障报修_变电设施', '公用事业类_电力_故障报修_独户停电',
       '公用事业类_电力_故障报修_电线高压线', '公用事业类_电力_故障报修_电表维修', '公用事业类_电力_故障报修_规模停电',
       '公用事业类_电力_故障报修_路灯供电', '公用事业类_电力_用电计费_分时用电', '公用事业类_电力_用电计费_电费争议',
       '公用事业类_电力_用电计费_缴费信息', '公用事业类_电力_用电计费_阶梯电价', '公用事业类_电力_电力服务_业务办理',
       '公用事业类_电力_电力服务_信息查询', '公用事业类_电力_电力服务_服务质量', '公用事业类_电力_立杆_无主立杆',
       '公用事业类_电力_违规用电_私排电线', '公用事业类_电力_违规用电_窃电举报', '公用事业类_电力_限电管理_节能用电',
       '公用事业类_邮政通信_其他_其他', '公用事业类_邮政通信_通信_业务办理', '公用事业类_邮政通信_通信_信息查询',
       '公用事业类_邮政通信_通信_故障报修', '公用事业类_邮政通信_通信_服务质量', '公用事业类_邮政通信_通信_电视线路',
       '公用事业类_邮政通信_通信_通信基站', '公用事业类_邮政通信_通信_通信线路', '公用事业类_邮政通信_邮政服务_其他业务',
       '公用事业类_邮政通信_邮政服务_快递服务', '公用事业类_邮政通信_邮政服务_服务质量',
       '公用事业类_邮政通信_邮政服务_邮件邮递', '公用事业类_邮政通信_邮政服务_邮政发行',
       '其他类_“12345”热线_12345热线_信息咨询', '其他类_“12345”热线_其他_其他',
       '其他类_“12345”热线_意见建议_业务范畴', '其他类_“12345”热线_意见建议_平台系统',
       '其他类_“12345”热线_意见建议_服务流程', '其他类_“12345”热线_批评承办部门_办事效率',
       '其他类_“12345”热线_批评承办部门_服务态度', '其他类_“12345”热线_批评承办部门_诉求未解决',
       '其他类_“12345”热线_批评承办部门_问题推脱', '其他类_“12345”热线_投诉批评_业务熟练',
       '其他类_“12345”热线_投诉批评_办事效率', '其他类_“12345”热线_投诉批评_承办推诿',
       '其他类_“12345”热线_投诉批评_服务态度', '其他类_“12345”热线_投诉批评_问题解决',
       '其他类_“12345”热线_表扬感谢_业务熟练', '其他类_“12345”热线_表扬感谢_办事效率',
       '其他类_“12345”热线_表扬感谢_服务态度', '其他类_“12345”热线_表扬感谢_问题解决',
       '其他类_“12345”热线_表扬承办部门_表扬感谢', '其他类_中央直属_中央直属_中央直属机构信息',
       '其他类_中央直属_其他_其他', '其他类_信访_事项办理_办理结果不满', '其他类_信访_事项办理_无答复意见',
       '其他类_信访_其他_其他', '其他类_信访_咨询信访渠道_咨询信访渠道', '其他类_信访_复查复核_催办',
       '其他类_信访_复查复核_流程指引', '其他类_信访_复查复核_结论不满', '其他类_信访_查询办理进度_查询办理进度',
       '其他类_其他_其他_其他', '其他类_其他_无效电话_外省市受理', '其他类_其他_无效电话_无效来电',
       '其他类_其他_无效电话_无明确诉求', '其他类_工青妇工作_工青妇工作_工青妇工作', '其他类_机关事务管理_其他_其他',
       '其他类_机关事务管理_机关事务管理_公务车', '其他类_机关事务管理_机关事务管理_机关事务信息',
       '其他类_机关事务管理_机关事务管理_机关事务工作', '其他类_残联_其他_其他', '其他类_残联_残联_助残信息',
       '其他类_残联_残联_无障碍工程', '其他类_残联_残联_残疾补助', '其他类_残联_残联_残疾鉴定',
       '其他类_残联_残联_残联工作', '其他类_涉港澳台_涉港澳台_涉台问题', '其他类_纪检监察_其他_其他',
       '其他类_纪检监察_政风行风_不作为', '其他类_纪检监察_政风行风_乱作为', '其他类_纪检监察_政风行风_办事推诿',
       '其他类_纪检监察_政风行风_慢作为', '其他类_纪检监察_政风行风_服务态度', '其他类_纪检监察_政风行风_违规收费',
       '其他类_纪检监察_纪检监查_作风问题', '其他类_纪检监察_纪检监查_失职渎职', '其他类_纪检监察_纪检监查_徇私枉法',
       '其他类_纪检监察_纪检监查_贪污贿赂', '其他类_部队_其他_其他', '其他类_部队_部队_涉军问题',
       '其他类_部队_部队_证件办理', '安全监管类_安全生产管理_其他_其他', '安全监管类_安全生产管理_安全生产_安全生产宣传',
       '安全监管类_安全生产管理_安全生产_安全生产监督', '安全监管类_安全生产管理_安全生产_应急救援',
       '安全监管类_安全生产管理_职业健康_职业健康', '安全监管类_质量技术监督_其他_其他',
       '安全监管类_质量技术监督_特种设备监察_人员资质', '安全监管类_质量技术监督_特种设备监察_安全监察',
       '安全监管类_质量技术监督_生产许可证管理_生产许可证管理', '安全监管类_质量技术监督_申诉举报_申诉举报',
       '安全监管类_质量技术监督_组织机构代码证_组织机构代码证', '安全监管类_质量技术监督_计量器具检测_计量器具检测',
       '安全监管类_质量技术监督_质量认证_质量认证', '安全监管类_食品药品安全_其他_其他',
       '安全监管类_食品药品安全_安全管理_保健品管理', '安全监管类_食品药品安全_安全管理_化妆品管理',
       '安全监管类_食品药品安全_安全管理_医疗器械管理', '安全监管类_食品药品安全_安全管理_药品安全',
       '安全监管类_食品药品安全_安全管理_资质认证', '安全监管类_食品药品安全_安全管理_食品安全',
       '安全监管类_食品药品安全_广告审批_广告审批', '建设交通类_交通港口_停车管理_公共停车场',
       '建设交通类_交通港口_停车管理_经营性停车场', '建设交通类_交通港口_公交巴士_服务规范',
       '建设交通类_交通港口_公交巴士_线路咨询', '建设交通类_交通港口_公交巴士_线路设置',
       '建设交通类_交通港口_公共交通用卡管理_交通卡', '建设交通类_交通港口_公共交通用卡管理_老年卡',
       '建设交通类_交通港口_其他_其他', '建设交通类_交通港口_出租车_服务规范', '建设交通类_交通港口_出租车_物品遗失',
       '建设交通类_交通港口_出租车_黑车', '建设交通类_交通港口_地铁客运_地铁建设',
       '建设交通类_交通港口_地铁客运_线路时间', '建设交通类_交通港口_汽车客运_票务',
       '建设交通类_交通港口_汽车客运_违规运营', '建设交通类_交通港口_汽车维修_4s店服务',
       '建设交通类_交通港口_汽车维修_故障牵引', '建设交通类_交通港口_港船客运_浦江轮渡',
       '建设交通类_交通港口_港船客运_省际轮渡', '建设交通类_交通港口_航空客运_航班延误',
       '建设交通类_交通港口_航空客运_飞机票务', '建设交通类_交通港口_货物运输_搬场公司',
       '建设交通类_交通港口_货物运输_航空运输', '建设交通类_交通港口_货物运输_集装箱运输',
       '建设交通类_交通港口_车牌拍卖_车牌拍卖', '建设交通类_交通港口_铁路客运_车次信息',
       '建设交通类_交通港口_铁路客运_铁路票务', '建设交通类_交通港口_驾驶培训_二次收费',
       '建设交通类_交通港口_驾驶培训_合同履行', '建设交通类_交通港口_驾驶培训_教练服务、培训规范',
       '建设交通类_住房保障_住房病虫害_白蚁', '建设交通类_住房保障_住房病虫害_鼠患',
       '建设交通类_住房保障_保障型住房_公租房', '建设交通类_住房保障_保障型住房_廉租房',
       '建设交通类_住房保障_保障型住房_经适房', '建设交通类_住房保障_其他_其他',
       '建设交通类_住房保障_市场交易管理_住房公积金', '建设交通类_住房保障_市场交易管理_房产信息变更',
       '建设交通类_住房保障_市场交易管理_房屋转让', '建设交通类_住房保障_征收管理_动迁政策',
       '建设交通类_住房保障_征收管理_司法拆迁', '建设交通类_住房保障_征收管理_土地征收',
       '建设交通类_住房保障_征收管理_拆迁安置', '建设交通类_住房保障_征收管理_违规拆迁',
       '建设交通类_住房保障_房屋权属_土地使用权', '建设交通类_住房保障_房屋权属_宅基地',
       '建设交通类_住房保障_房屋权属_居改非', '建设交通类_住房保障_房屋权属_房屋产权',
       '建设交通类_住房保障_房屋权属_租赁管理', '建设交通类_住房保障_旧房改造_平改坡',
       '建设交通类_住房保障_旧房改造_房屋老化', '建设交通类_住房保障_旧房改造_管道改建',
       '建设交通类_住房保障_旧房改造_结构改建', '建设交通类_住房保障_物业服务管理_业委会',
       '建设交通类_住房保障_物业服务管理_公房管理', '建设交通类_住房保障_物业服务管理_垃圾清理',
       '建设交通类_住房保障_物业服务管理_房屋结构破坏', '建设交通类_住房保障_物业服务管理_物业安保',
       '建设交通类_住房保障_物业服务管理_纠纷协调', '建设交通类_住房保障_物业服务管理_维修基金',
       '建设交通类_住房保障_物业服务管理_维修添置', '建设交通类_住房保障_物业服务管理_绿化维护',
       '建设交通类_住房保障_物业服务管理_群租现象', '建设交通类_口岸_其他_其他', '建设交通类_口岸_口岸信息_口岸信息',
       '建设交通类_土地规划_其他_其他', '建设交通类_土地规划_土地资源管理_地面沉降',
       '建设交通类_土地规划_城市建设用地管理_历史建筑', '建设交通类_土地规划_城市建设用地管理_档案管理',
       '建设交通类_土地规划_城市建设用地管理_测绘管理', '建设交通类_土地规划_城市建设用地管理_规划设计',
       '建设交通类_土地规划_建设管理_建筑间距及日照', '建设交通类_土地规划_建设管理_许可证管理',
       '建设交通类_土地规划_非城市建设用地管理_农村建设用地管理', '建设交通类_土地规划_非城市建设用地管理_耕地管理',
       '建设交通类_城乡建设_公共道路_天桥', '建设交通类_城乡建设_公共道路_安全隐患',
       '建设交通类_城乡建设_公共道路_桥梁通行', '建设交通类_城乡建设_公共道路_路状路况',
       '建设交通类_城乡建设_公共道路_道监管理', '建设交通类_城乡建设_公共道路_道路结冰积雪',
       '建设交通类_城乡建设_公共道路_隧道', '建设交通类_城乡建设_公共道路_高架高速', '建设交通类_城乡建设_其他_其他',
       '建设交通类_城乡建设_工程管理_危房', '建设交通类_城乡建设_工程管理_工程质量',
       '建设交通类_城乡建设_工程管理_市政工程', '建设交通类_城乡建设_工程管理_建筑工程欠薪',
       '建设交通类_城乡建设_工程管理_施工规范', '建设交通类_城乡建设_工程管理_筑材光污染',
       '建设交通类_城乡建设_燃气行业监管_燃气行业', '建设交通类_城乡建设_违法建筑_违法建筑',
       '建设交通类_城乡建设_道路维护_断头路', '建设交通类_城乡建设_道路维护_设备添置',
       '建设交通类_城乡建设_道路维护_路标路牌', '建设交通类_城乡建设_道路维护_路面养护',
       '建设交通类_城乡建设_道路维护_路面拓宽', '建设交通类_城乡建设_道路维护_道路改建',
       '建设交通类_城乡建设_道路维护_隔离护栏', '建设交通类_民防_其他_其他', '建设交通类_民防_民防工程_地下室管理',
       '建设交通类_民防_民防工程_工程规划', '建设交通类_民防_民防工程_民防警报', '建设交通类_民防_防灾抗灾_地震',
       '建设交通类_民防_防灾抗灾_抗台防汛', '建设交通类_水务_其他_其他', '建设交通类_水务_排水排污管理_下水道养护',
       '建设交通类_水务_排水排污管理_不明归属井盖', '建设交通类_水务_排水排污管理_市政道路积水',
       '建设交通类_水务_排水排污管理_排水工程', '建设交通类_水务_排水排污管理_排污许可证',
       '建设交通类_水务_排水排污管理_水务工程建设', '建设交通类_水务_排水排污管理_污水冒溢',
       '建设交通类_水务_水资源管理_农业水利', '建设交通类_水务_水资源管理_地表地下水',
       '建设交通类_水务_水资源管理_水资源信息', '建设交通类_水务_水资源管理_河道水质',
       '建设交通类_水务_水资源管理_非上水供水', '建设交通类_水务_河道流域管理_水上漂浮物',
       '建设交通类_水务_河道流域管理_河上搭建', '建设交通类_水务_河道流域管理_河道养护',
       '建设交通类_水务_河道流域管理_河道疏通', '建设交通类_水务_河道流域管理_生态保护',
       '建设交通类_水务_河道流域管理_防汛抗灾', '建设交通类_环境保护_其他_其他',
       '建设交通类_环境保护_夜间施工许可_夜间施工的现象', '建设交通类_环境保护_夜间施工许可_夜间施工证的审批',
       '建设交通类_环境保护_污染_光污染', '建设交通类_环境保护_污染_化工污染', '建设交通类_环境保护_污染_噪音污染',
       '建设交通类_环境保护_污染_固废危废', '建设交通类_环境保护_污染_废气污染', '建设交通类_环境保护_污染_水污染',
       '建设交通类_环境保护_污染_油烟扰民', '建设交通类_环境保护_污染_焚烧垃圾', '建设交通类_环境保护_污染_电磁辐射污染',
       '建设交通类_环境保护_污染_空气污染', '建设交通类_环境保护_环保设备设施_环保设备设施',
       '建设交通类_绿化市容_其他_其他', '建设交通类_绿化市容_动植园林_信息查询', '建设交通类_绿化市容_动植园林_公园设施',
       '建设交通类_绿化市容_动植园林_管理规范', '建设交通类_绿化市容_动植园林_野生动植物',
       '建设交通类_绿化市容_动植园林_防治检疫', '建设交通类_绿化市容_城市管理_城管执法',
       '建设交通类_绿化市容_城市管理_无证夜间施工', '建设交通类_绿化市容_城市管理_无证设摊',
       '建设交通类_绿化市容_城市管理_跨门经营', '建设交通类_绿化市容_市容市貌_各类亭、岗',
       '建设交通类_绿化市容_市容市貌_广告张贴', '建设交通类_绿化市容_市容市貌_广告牌景观灯',
       '建设交通类_绿化市容_环境卫生_倾倒渣土', '建设交通类_绿化市容_环境卫生_单位垃圾管理',
       '建设交通类_绿化市容_环境卫生_卫生配套设施', '建设交通类_绿化市容_环境卫生_收费规范管理',
       '建设交通类_绿化市容_环境卫生_特定污染物处治', '建设交通类_绿化市容_环境卫生_环卫作业规范',
       '建设交通类_绿化市容_环境卫生_环境垃圾', '建设交通类_绿化市容_环境卫生_粪便冒溢',
       '建设交通类_绿化市容_环境卫生_车辆清洗管理', '建设交通类_绿化市容_环境卫生_道路扬尘',
       '建设交通类_绿化市容_禽类饲养_农村地区家禽饲养', '建设交通类_绿化市容_禽类饲养_城区家禽饲养',
       '建设交通类_绿化市容_禽类饲养_无证养鸽', '建设交通类_绿化市容_禽类饲养_观赏鸟',
       '建设交通类_绿化市容_绿地绿化_占绿毁绿', '建设交通类_绿化市容_绿地绿化_古树名木',
       '建设交通类_绿化市容_绿地绿化_增添绿化', '建设交通类_绿化市容_绿地绿化_林业管理',
       '建设交通类_绿化市容_绿地绿化_移植修剪', '建设交通类_绿化市容_绿地绿化_绿化护栏',
       '建设交通类_绿化市容_绿地绿化_绿化着火', '建设交通类_绿化市容_绿地绿化_行道树等树木倾斜倒伏',
       '建设交通类_绿化市容_绿地绿化_非法经营利用', '社会团体类_其他_其他_其他', '社会团体类_其他_其他社团_动物保护机构',
       '社会团体类_其他_其他社团_民间慈善', '社会团体类_其他_其他社团_红十字会', '社会团体类_团市委_其他_其他',
       '社会团体类_团市委_团市委_团委工作', '社会团体类_团市委_团市委_青少年保护', '社会团体类_妇联_其他_其他',
       '社会团体类_妇联_妇联_妇女健康', '社会团体类_妇联_妇联_妇女权益维护', '社会团体类_妇联_妇联_妇联工作',
       '社会团体类_总工会_其他_其他', '社会团体类_总工会_总工会_劳模', '社会团体类_总工会_总工会_工会工作',
       '社会团体类_总工会_总工会_职工权益', '社会管理类_人力保障_其他_其他',
       '社会管理类_人力保障_劳动人事制度_公务员管理', '社会管理类_人力保障_劳动人事制度_养老金调整',
       '社会管理类_人力保障_劳动人事制度_劳务派遣', '社会管理类_人力保障_劳动人事制度_工作调动',
       '社会管理类_人力保障_劳动人事制度_工时假期', '社会管理类_人力保障_劳动人事制度_工资奖金',
       '社会管理类_人力保障_劳动人事制度_工龄核算', '社会管理类_人力保障_劳动人事制度_职称评定',
       '社会管理类_人力保障_劳动人事制度_退休离休', '社会管理类_人力保障_劳动保护_合同纠纷',
       '社会管理类_人力保障_劳动保护_拖欠工资', '社会管理类_人力保障_劳动保护_最低工资',
       '社会管理类_人力保障_劳动保护_特殊工种', '社会管理类_人力保障_劳动保护_用工规范',
       '社会管理类_人力保障_劳动保护_经济补偿', '社会管理类_人力保障_医疗医保_医保定点',
       '社会管理类_人力保障_医疗医保_医保报销', '社会管理类_人力保障_医疗医保_医保政策咨询',
       '社会管理类_人力保障_医疗医保_医保药品目录', '社会管理类_人力保障_就业创业_人才引进',
       '社会管理类_人力保障_就业创业_创业扶持', '社会管理类_人力保障_就业创业_劳务中介',
       '社会管理类_人力保障_就业创业_就业信息', '社会管理类_人力保障_就业创业_职业培训',
       '社会管理类_人力保障_就业创业_转业干部', '社会管理类_人力保障_工伤_劳动能力鉴定',
       '社会管理类_人力保障_工伤_工伤认定', '社会管理类_人力保障_津贴补助_丧葬补助金',
       '社会管理类_人力保障_津贴补助_事业单位共享费', '社会管理类_人力保障_津贴补助_其他补助',
       '社会管理类_人力保障_津贴补助_培训补贴', '社会管理类_人力保障_津贴补助_就业补贴',
       '社会管理类_人力保障_津贴补助_高温津贴', '社会管理类_人力保障_社会保险_养老保险',
       '社会管理类_人力保障_社会保险_医疗保险', '社会管理类_人力保障_社会保险_失业保险',
       '社会管理类_人力保障_社会保险_工伤保险', '社会管理类_人力保障_社会保险_生育保险',
       '社会管理类_人力保障_社保信息_受理网点', '社会管理类_人力保障_社保信息_居住证',
       '社会管理类_人力保障_社保信息_社保卡', '社会管理类_人力保障_社保信息_账户信息',
       '社会管理类_人力保障_社保信息_退休证', '社会管理类_侨务_侨务工作_侨胞投资扶持', '社会管理类_侨务_其他_其他',
       '社会管理类_合作交流_其他_其他', '社会管理类_合作交流_合作交流_合作交流', '社会管理类_外事_其他_其他',
       '社会管理类_外事_外事工作_国际会议', '社会管理类_外事_外事工作_外宾来访', '社会管理类_外事_外事工作_领事馆',
       '社会管理类_工商(消保)_其他_其他', '社会管理类_工商(消保)_商业服务_各类服务卡券',
       '社会管理类_工商(消保)_商业服务_售后服务', '社会管理类_工商(消保)_商业服务_商品质量',
       '社会管理类_工商(消保)_商业服务_抵押拍卖', '社会管理类_工商(消保)_商业服务_服务质量',
       '社会管理类_工商(消保)_商业服务_活禽交易', '社会管理类_工商(消保)_商业服务_电视购物',
       '社会管理类_工商(消保)_商业服务_网上购物', '社会管理类_工商(消保)_商业服务_菜场经营',
       '社会管理类_工商(消保)_商业服务_超市卖场', '社会管理类_工商(消保)_工商信息_商品维修网点',
       '社会管理类_工商(消保)_工商信息_商品销售信息', '社会管理类_工商(消保)_工商信息_维权咨询',
       '社会管理类_工商(消保)_工商监管_其他许可', '社会管理类_工商(消保)_工商监管_合同监管',
       '社会管理类_工商(消保)_工商监管_商标包装管理', '社会管理类_工商(消保)_工商监管_审验登记',
       '社会管理类_工商(消保)_工商监管_广告监督', '社会管理类_工商(消保)_工商监管_食品安全流通',
       '社会管理类_工商(消保)_违法经营_不正当经营', '社会管理类_工商(消保)_违法经营_假冒伪劣',
       '社会管理类_工商(消保)_违法经营_无照经营', '社会管理类_工商(消保)_违法经营_违法传销',
       '社会管理类_旅游_其他_其他', '社会管理类_旅游_旅游信息_上海旅游节', '社会管理类_旅游_旅游信息_旅行社信息',
       '社会管理类_旅游_旅游信息_景点投诉', '社会管理类_旅游_旅游信息_景点资讯', '社会管理类_旅游_旅游纠纷_强制消费',
       '社会管理类_旅游_旅游纠纷_旅游合同纠纷', '社会管理类_旅游_服务质量_导游服务', '社会管理类_旅游_服务质量_旅游交通',
       '社会管理类_旅游_服务质量_旅行社服务', '社会管理类_旅游_服务质量_景点服务质量',
       '社会管理类_旅游_服务质量_酒店质量', '社会管理类_旅游_服务质量_餐饮服务', '社会管理类_档案_个人档案_个人档案',
       '社会管理类_档案_其他_其他', '社会管理类_档案_国有档案_国有企业', '社会管理类_档案_国有档案_政府机构',
       '社会管理类_档案_国有档案_服务质量', '社会管理类_档案_国有档案_档案公开', '社会管理类_民政_优抚安置_军人伤残优抚',
       '社会管理类_民政_优抚安置_支内知青补贴', '社会管理类_民政_优抚安置_烈士家属优抚',
       '社会管理类_民政_优抚安置_退伍复员', '社会管理类_民政_其他_其他', '社会管理类_民政_养老保障_孤寡空巢老人关怀',
       '社会管理类_民政_养老保障_居家养老', '社会管理类_民政_养老保障_机构养老', '社会管理类_民政_婚姻管理_单身证明',
       '社会管理类_民政_婚姻管理_离婚办理', '社会管理类_民政_婚姻管理_结婚登记', '社会管理类_民政_慈善福利_公益彩票',
       '社会管理类_民政_慈善福利_孤儿收养', '社会管理类_民政_慈善福利_慈善募捐', '社会管理类_民政_慈善福利_福利企业扶持',
       '社会管理类_民政_慈善福利_福利设施', '社会管理类_民政_救灾赈灾_救灾物资分配', '社会管理类_民政_救灾赈灾_社会捐献',
       '社会管理类_民政_殡葬事业_墓园管理', '社会管理类_民政_殡葬事业_殡葬服务', '社会管理类_民政_生活保障_扶贫帮困',
       '社会管理类_民政_生活保障_最低生活保障', '社会管理类_民政_生活保障_流浪救助',
       '社会管理类_民政_生活保障_经济状况审核', '社会管理类_民政_社区家政_家政服务',
       '社会管理类_民政_社区家政_社区服务网点', '社会管理类_民政_社区家政_社工管理', '社会管理类_民族宗教_其他_其他',
       '社会管理类_民族宗教_宗教信仰_宗教信仰', '社会管理类_民族宗教_寺庙管理_人员管理',
       '社会管理类_民族宗教_寺庙管理_寺庙用品', '社会管理类_民族宗教_寺庙管理_收费标准',
       '社会管理类_民族宗教_寺庙管理_秩序维护', '社会管理类_知识产权_其他_其他', '社会管理类_知识产权_知识产权_侵权行为',
       '社会管理类_知识产权_知识产权_服务质量', '社会管理类_知识产权_知识产权_申请专利',
       '社会管理类_社区综合服务_社区综合治理_党群工作', '社会管理类_社区综合服务_社区综合治理_平安建设',
       '社会管理类_社区综合服务_社区综合治理_群体事件', '社会管理类_社区综合服务_社区综合治理_重大传染病疫情',
       '科教文卫类_体育_体育管理_体育彩票', '科教文卫类_体育_体育管理_信鸽', '科教文卫类_体育_体育管理_场地设施',
       '科教文卫类_体育_体育管理_群众锻炼', '科教文卫类_体育_体育管理_赛事管理', '科教文卫类_体育_体育管理_队伍建设',
       '科教文卫类_体育_体育管理_青少年锻炼', '科教文卫类_体育_其他_其他', '科教文卫类_卫生计生_人口计生_业务办理',
       '科教文卫类_卫生计生_人口计生_优生优育', '科教文卫类_卫生计生_人口计生_再生育政策',
       '科教文卫类_卫生计生_人口计生_婚前检查', '科教文卫类_卫生计生_人口计生_独生子女补贴',
       '科教文卫类_卫生计生_人口计生_网点查询', '科教文卫类_卫生计生_人口计生_计生宣传',
       '科教文卫类_卫生计生_人口计生_超生处罚', '科教文卫类_卫生计生_人口计生_避孕节育', '科教文卫类_卫生计生_其他_其他',
       '科教文卫类_卫生计生_医疗服务_住院病床', '科教文卫类_卫生计生_医疗服务_医疗设备',
       '科教文卫类_卫生计生_医疗服务_护工护理', '科教文卫类_卫生计生_医疗服务_排队挂号',
       '科教文卫类_卫生计生_医疗服务_服务态度', '科教文卫类_卫生计生_医疗服务_用药规范',
       '科教文卫类_卫生计生_医院监管_医疗纠纷', '科教文卫类_卫生计生_医院监管_职业资质许可',
       '科教文卫类_卫生计生_医院监管_非法行医', '科教文卫类_卫生计生_卫生信息_医药卫生知识',
       '科教文卫类_卫生计生_卫生信息_医院医疗信息', '科教文卫类_卫生计生_卫生信息_疾控防疫',
       '科教文卫类_卫生计生_卫生信息_社区卫生服务', '科教文卫类_卫生计生_特定医疗机构_临终关怀',
       '科教文卫类_卫生计生_特定医疗机构_体检中心', '科教文卫类_卫生计生_特定医疗机构_器官捐赠',
       '科教文卫类_卫生计生_特定医疗机构_妇婴保健', '科教文卫类_卫生计生_特定医疗机构_采血机构',
       '科教文卫类_教育_入学入托_入学政策', '科教文卫类_教育_入学入托_办理转学', '科教文卫类_教育_入学入托_外地户口入学',
       '科教文卫类_教育_入学入托_居住证', '科教文卫类_教育_其他_其他', '科教文卫类_教育_学校管理_学籍管理',
       '科教文卫类_教育_学校管理_收费问题', '科教文卫类_教育_学校管理_教师素质', '科教文卫类_教育_学校管理_教育改革',
       '科教文卫类_教育_学校管理_教育质量', '科教文卫类_教育_学校管理_校园设施', '科教文卫类_教育_学校管理_费用减免',
       '科教文卫类_教育_学生负担_作业负担', '科教文卫类_教育_学生负担_超时补课', '科教文卫类_教育_招生考试_中考',
       '科教文卫类_教育_招生考试_居住证', '科教文卫类_教育_招生考试_成人教育', '科教文卫类_教育_招生考试_研究生考试',
       '科教文卫类_教育_招生考试_职业技能考试', '科教文卫类_教育_招生考试_高考', '科教文卫类_文广影视_其他_其他',
       '科教文卫类_文广影视_广播影视_东方有线', '科教文卫类_文广影视_广播影视_广播',
       '科教文卫类_文广影视_广播影视_影视节目', '科教文卫类_文广影视_广播影视_网络视听',
       '科教文卫类_文广影视_文化产业管理_文化场所信息', '科教文卫类_文广影视_文化产业管理_文化活动',
       '科教文卫类_文广影视_文化产业管理_文化社团', '科教文卫类_文广影视_文化产业管理_游戏动漫',
       '科教文卫类_文广影视_文化产业管理_许可审批', '科教文卫类_文广影视_文化遗产保护_文物保护管理',
       '科教文卫类_文广影视_文化遗产保护_非物质文化遗产', '科教文卫类_新闻出版_其他_其他',
       '科教文卫类_新闻出版_刊物市场监管_刊社信息', '科教文卫类_新闻出版_刊物市场监管_图书刊物出版',
       '科教文卫类_新闻出版_刊物市场监管_报刊印刷发行', '科教文卫类_新闻出版_刊物市场监管_科教书',
       '科教文卫类_新闻出版_新闻报道_新闻内容', '科教文卫类_新闻出版_新闻报道_记者证管理', '科教文卫类_科技_其他_其他',
       '科教文卫类_科技_科技开发_发明建议', '科教文卫类_科技_科技开发_自主研发', '科教文卫类_科技_科技管理_科技改革',
       '科教文卫类_科技_科技管理_高新技术扶持', '经济综合类_农业_其他_其他', '经济综合类_农业_农业生产_农产品安全',
       '经济综合类_农业_农业生产_农副业生产', '经济综合类_农业_农业生产_土地建设与保护',
       '经济综合类_农业_农业生产_滥砍滥捕', '经济综合类_农业_农业生产_畜牧养殖',
       '经济综合类_农业_农业生产_转基因产品（有机食品）', '经济综合类_农业_农民负担_农业补贴',
       '经济综合类_农业_农民负担_国家收购', '经济综合类_农业_动植物防疫_兽医药管理',
       '经济综合类_农业_动植物防疫_疫情防疫', '经济综合类_农业_动植物防疫_防疫补贴', '经济综合类_农业_村队管理_吸劳养老',
       '经济综合类_农业_村队管理_土地承包', '经济综合类_农业_村队管理_征地补偿', '经济综合类_农业_村队管理_撤制村队',
       '经济综合类_农业_村队管理_集体资产管理', '经济综合类_发展改革_价格调控_公共事业',
       '经济综合类_发展改革_价格调控_医疗服务', '经济综合类_发展改革_价格调控_市场物价',
       '经济综合类_发展改革_价格调控_物业管理收费', '经济综合类_发展改革_其他_其他',
       '经济综合类_发展改革_改革规划_医疗改革', '经济综合类_发展改革_改革规划_新能源汽车',
       '经济综合类_发展改革_改革规划_能源发展', '经济综合类_商务_其他_其他', '经济综合类_商务_商务_商贸信息',
       '经济综合类_商务_商务_市场供应', '经济综合类_商务_商务_烟草酒类', '经济综合类_国资_其他_其他',
       '经济综合类_国资_国资_企业改革', '经济综合类_国资_国资_企业解困', '经济综合类_国资_国资_国企管理',
       '经济综合类_国资_国资_国有资产流失', '经济综合类_国资_国资_投资决策', '经济综合类_审计_其他_其他',
       '经济综合类_审计_审计工作_审计工作', '经济综合类_税务_其他_其他', '经济综合类_税务_税收政策_个人所得税',
       '经济综合类_税务_税收政策_企业征税', '经济综合类_税务_税收政策_其他税种', '经济综合类_税务_税收政策_房产交易税',
       '经济综合类_税务_税收政策_营业税改增值税', '经济综合类_税务_税收管理_假发票', '经济综合类_税务_税收管理_偷税漏税',
       '经济综合类_税务_税收管理_虚开发票', '经济综合类_经济信息化_其他_其他',
       '经济综合类_经济信息化_经济信息化_城市信息化', '经济综合类_经济信息化_经济信息化_循环再利用',
       '经济综合类_经济信息化_经济信息化_法人一证通', '经济综合类_经济信息化_经济信息化_节能减排',
       '经济综合类_统计_其他_其他', '经济综合类_统计_统计工作_资质证书', '经济综合类_财政_其他_其他',
       '经济综合类_财政_财政工作_财会考证', '经济综合类_财政_财政工作_财政票据', '经济综合类_金融_其他_其他',
       '经济综合类_金融_金融_保险监管', '经济综合类_金融_金融_证券监管', '经济综合类_金融_金融_资产评估',
       '经济综合类_金融_金融_金融资讯', '经济综合类_金融_金融_银行监管']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    # def _create_examples(self, lines, set_type):
    #     """Creates examples for the training and dev sets."""
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         guid = "%s-%s" % (set_type, i)
    #         text_a = line[0]
    #         text_b = line[1]
    #         label = line[-1]
    #         examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #     return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
