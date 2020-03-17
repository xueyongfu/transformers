

import argparse
import logging
import sys
import unittest
from unittest.mock import patch

import run_generation
import run_glue
import run_squad


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


class ExamplesTests(unittest.TestCase):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = [
            "run_glue.py",
            "--data_dir=./examples/tests_samples/MRPC/",
            "--task_name=mrpc",
            "--do_train",
            "--do_eval",
            "--output_dir=./examples/tests_samples/temp_dir",
            "--per_gpu_train_batch_size=2",
            "--per_gpu_eval_batch_size=1",
            "--learning_rate=1e-4",
            "--max_steps=10",
            "--warmup_steps=2",
            "--overwrite_output_dir",
            "--seed=42",
        ]
        model_type, model_name = ("--model_type=bert", "--model_name_or_path=bert-base-uncased")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_glue.main()
            for value in result.values():
                self.assertGreaterEqual(value, 0.75)

    def test_run_squad(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = [
            "run_squad.py",
            "--data_dir=./examples/tests_samples/SQUAD",
            "--model_name=bert-base-uncased",
            "--output_dir=./examples/tests_samples/temp_dir",
            "--max_steps=10",
            "--warmup_steps=2",
            "--do_train",
            "--do_eval",
            "--version_2_with_negative",
            "--learning_rate=2e-4",
            "--per_gpu_train_batch_size=2",
            "--per_gpu_eval_batch_size=1",
            "--overwrite_output_dir",
            "--seed=42",
        ]
        model_type, model_name = ("--model_type=bert", "--model_name_or_path=bert-base-uncased")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_squad.main()
            self.assertGreaterEqual(result["f1"], 30)
            self.assertGreaterEqual(result["exact"], 30)

    def test_generation(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = ["run_generation.py", "--prompt=Hello", "--length=10", "--seed=42"]
        model_type, model_name = ("--model_type=openai-gpt", "--model_name_or_path=openai-gpt")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)
