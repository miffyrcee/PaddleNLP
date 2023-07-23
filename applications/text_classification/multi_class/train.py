# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from paddlenlp.utils.log import logger


def preprocess_function(examples, tokenizer, max_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
    """
    result = tokenizer(examples["text"], max_length=max_length, truncation=True)
    if not is_test:
        result["labels"] = np.array([examples["label"]], dtype="int64")
    return result


def read_local_dataset(path, label2id=None, is_test=False):
    """
    Read dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                if is_test:
                    sentence = line.strip()
                    yield {"text": sentence}
                else:
                    items = line.strip().split("\t")
                    yield {"text": items[0], "label": label2id[items[1]]}
            except:
                print(f"--*--line:--*--{line}--*--")


def log_metrics_debug(output, id2label, dev_ds, bad_case_path):
    """
    Log metrics in debug mode.
    """
    predictions, label_ids, metrics = output
    pred_ids = np.argmax(predictions, axis=-1)
    logger.info("-----Evaluate model-------")
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(metrics["test_accuracy"] * 100))
    logger.info(
        "Macro average | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            metrics["test_macro avg"]["precision"] * 100,
            metrics["test_macro avg"]["recall"] * 100,
            metrics["test_macro avg"]["f1-score"] * 100,
        )
    )
    for i in id2label:
        l = id2label[i]
        logger.info("Class name: {}".format(l))
        i = "test_" + str(i)
        if i in metrics:
            logger.info(
                "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                    metrics[i]["support"],
                    100 * metrics[i]["support"] / len(dev_ds),
                    metrics[i]["precision"] * 100,
                    metrics[i]["recall"] * 100,
                    metrics[i]["f1-score"] * 100,
                )
            )
        else:
            logger.info("Evaluation examples in dev dataset: 0 (0%)")
        logger.info("----------------------------")

    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i, (p, l) in enumerate(zip(pred_ids, label_ids)):
            p, l = int(p), int(l)
            if p != l:
                f.write(dev_ds.data[i]["text"] + "\t" + id2label[l] + "\t" + id2label[p] + "\n")

    logger.info("Bad case in dev dataset saved in {}".format(bad_case_path))


from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    EarlyStoppingCallback,
    PdArgumentParser,
    Trainer,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    export_model,
)
from paddlenlp.utils.log import logger

SUPPORTED_MODELS = [
    "ernie-1.0-large-zh-cw",
    "ernie-1.0-base-zh-cw",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-base-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-nano-zh",
    "ernie-3.0-tiny-base-v2-zh",
    "ernie-3.0-tiny-medium-v2-zh",
    "ernie-3.0-tiny-micro-v2-zh",
    "ernie-3.0-tiny-mini-v2-zh",
    "ernie-3.0-tiny-nano-v2-zh ",
    "ernie-3.0-tiny-pico-v2-zh",
    "ernie-2.0-large-en",
    "ernie-2.0-base-en",
    "ernie-3.0-tiny-mini-v2-en",
    "ernie-m-base",
    "ernie-m-large",
]


# yapf: disable
@dataclass
class DataArguments:
    max_length: int = field(default=128, metadata={"help": "Maximum number of tokens for the model."})
    early_stopping: bool = field(default=False, metadata={"help": "Whether apply early stopping strategy."})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})
    debug: bool = field(default=False, metadata={"help": "Whether choose debug mode."})
    train_path: str = field(default='./data/train.txt', metadata={"help": "Train dataset file path."})
    dev_path: str = field(default='./data/dev.txt', metadata={"help": "Dev dataset file path."})
    test_path: str = field(default='./data/dev.txt', metadata={"help": "Test dataset file path."})
    label_path: str = field(default='./data/label.txt', metadata={"help": "Label file path."})
    bad_case_path: str = field(default='./data/bad_case.txt', metadata={"help": "Bad case file path."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-tiny-medium-v2-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to directory to store the exported inference model."})
# yapf: enable


class FakeClass:
    def __init__(self, xxx):
        self.set(xxx)

    def set(self, xxx):
        for k, v in xxx.items():
            setattr(self, k, v)

    def print_config(self, *args, **kwargs):
        print("xx")


def main():
    """
    Training a binary or multi classification model
    """
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # model_args, data_args, training_args = FakeClass(config), FakeClass(config), FakeClass(config)

    if training_args.do_compress:
        training_args.strategy = "dynabert"
    if training_args.do_train or training_args.do_compress:
        training_args.print_config(model_args, "Model")
        training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Define id2label
    id2label = {}
    label2id = {}
    with open(data_args.label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            id2label[i] = l
            label2id[l] = i

    # Define model & tokenizer
    if os.path.isdir(model_args.model_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, label2id=label2id, id2label=id2label
        )
    elif model_args.model_name_or_path in SUPPORTED_MODELS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_classes=len(label2id), label2id=label2id, id2label=id2label
        )
    else:
        raise ValueError(
            f"{model_args.model_name_or_path} is not a supported model type. Either use a local model path or select a model from {SUPPORTED_MODELS}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load and preprocess dataset
    train_ds = load_dataset(read_local_dataset, path=data_args.train_path, label2id=label2id, lazy=False)
    dev_ds = load_dataset(read_local_dataset, path=data_args.dev_path, label2id=label2id, lazy=False)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_length=data_args.max_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    if data_args.debug:
        test_ds = load_dataset(read_local_dataset, path=data_args.test_path, label2id=label2id, lazy=False)
        test_ds = test_ds.map(trans_func)

    # Define the metric function.
    def compute_metrics(eval_preds):
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        for average in ["micro", "macro"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=eval_preds.label_ids, y_pred=pred_ids, average=average
            )
            metrics[f"{average}_precision"] = precision
            metrics[f"{average}_recall"] = recall
            metrics[f"{average}_f1"] = f1
        return metrics

    def compute_metrics_debug(eval_preds):
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        metrics = classification_report(eval_preds.label_ids, pred_ids, output_dict=True)
        return metrics

    # Define the early-stopping callback.
    if data_args.early_stopping:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
    else:
        callbacks = None

    # Define Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=paddle.nn.loss.CrossEntropyLoss(),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_debug if data_args.debug else compute_metrics,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        for checkpoint_path in Path(training_args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_path)

    # Evaluate and tests model
    if training_args.do_eval:
        if data_args.debug:
            output = trainer.predict(test_ds)
            log_metrics_debug(output, id2label, test_ds, data_args.bad_case_path)
        else:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)

    # export inference model
    if training_args.do_export:
        if model.init_config["init_class"] in ["ErnieMForSequenceClassification"]:
            input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)
        tokenizer.save_pretrained(model_args.export_model_dir)
        id2label_file = os.path.join(model_args.export_model_dir, "id2label.json")
        with open(id2label_file, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False)
            logger.info(f"id2label file saved in {id2label_file}")

    # compress
    if training_args.do_compress:
        trainer.compress()
        for width_mult in training_args.width_mult_list:
            pruned_infer_model_dir = os.path.join(training_args.output_dir, "width_mult_" + str(round(width_mult, 2)))
            tokenizer.save_pretrained(pruned_infer_model_dir)
            id2label_file = os.path.join(pruned_infer_model_dir, "id2label.json")
            with open(id2label_file, "w", encoding="utf-8") as f:
                json.dump(id2label, f, ensure_ascii=False)
                logger.info(f"id2label file saved in {id2label_file}")

    for path in Path(training_args.output_dir).glob("runs"):
        shutil.rmtree(path)


if __name__ == "__main__":
    import sys
    import copy 

    sys.argv.extend(
        "--do_train --do_eval --do_export --model_name_or_path ernie-3.0-tiny-micro-v2-zh --output_dir checkpoint --device gpu --num_train_epochs 100 --early_stopping True --early_stopping_patience 5 --learning_rate 3e-3 --max_length 128 --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --metric_for_best_model accuracy --load_best_model_at_end --logging_steps 5 --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1".split()
    )
    for i in copy.copy(sys.argv):
        if i.find('-f') > -1:
            sys.argv.remove(i)
        elif i.find('jupyter') > -1:
            sys.argv.remove(i)
    main()

