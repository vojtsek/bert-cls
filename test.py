import transformers
from tensorflow.keras import preprocessing
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
import argparse

from czert import CZERT
from train import SEQ_LEN, load_dataset
import os
import yaml

def predict_with_model(sentence, model, tokenizer):
    bert_input = tokenizer(sentence, truncation=True, max_length=SEQ_LEN, padding='max_length')
    bert_input = {
        'input_ids': tf.constant(bert_input['input_ids']),
        'token_type_ids': bert_input['token_type_ids'],
        'attention_mask': tf.constant(bert_input['attention_mask'])
    }
    output = tf.argmax(model(bert_input), axis=-1)
    return output.numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--weights')
    parser.add_argument('--test_file')
    labels = []
    for root, dirs, files in os.walk('data'):
        labels.extend(dirs)
    labels = sorted(labels)

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = CZERT(args.model_dir, num_labels=578)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.load_weights(args.weights)
    with open(args.test_file, 'rt') as fd:
        data = yaml.safe_load(fd)
    correct = []
    for story in data['stories']:
        for step in story['steps']:
            if 'user' not in step:
                continue
            predicted = predict_with_model([step['user'] + '\n'], model, tokenizer)
            step['predicted'] = labels[predicted]
            correct.append(int(labels[predicted] == step['intent']))
    with open('predicted.yaml', 'wt') as fd:
        yaml.dump(data, fd)
    print(sum(correct)/len(correct), len(correct), sum(correct))

