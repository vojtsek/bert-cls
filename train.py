import torch
# from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import transformers
from tensorflow.keras import preprocessing
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
import argparse

from czert import CZERT
import tensorflow_datasets as tfds
SEQ_LEN=100
batch_size = 4


def predict_with_model(sentence, model, tokenizer):
    bert_input = tokenizer(sentence, truncation=True, max_length=SEQ_LEN, padding='max_length')
    bert_input = {
        'input_ids': tf.constant(bert_input['input_ids']),
        'token_type_ids': bert_input['token_type_ids'],
        'attention_mask': tf.constant(bert_input['attention_mask'])
    }
    output = tf.argmax(model(bert_input))
    return output.numpy()


def load_dataset(path):
    return preprocessing.text_dataset_from_directory(path,
                                                     labels='inferred',
                                                     label_mode='int', # or categorical
                                                     batch_size=1,
                                                     shuffle=True
                                                    )


def encode_examples(ds, tokenizer: transformers.PreTrainedTokenizer, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for utterance, label in tfds.as_numpy(ds):
        bert_input = tokenizer(utterance[0].decode(), truncation=True, max_length=SEQ_LEN, padding='max_length')

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        {'input_ids': tf.constant(input_ids_list),
         'token_type_ids': tf.constant(token_type_ids_list),
         'attention_mask': tf.constant(attention_mask_list),
         'label': label_list
        }).map(lambda example:({
            'input_ids': example['input_ids'],
            'token_type_ids': example['token_type_ids'],
            'attention_mask': example['attention_mask']
            }, example['label']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    # (ds_train, ds_test), ds_info = tfds.load('imdb_reviews',
    #                                          split=(tfds.Split.TRAIN, tfds.Split.TEST),
    #                                          as_supervised=True,
    #                                          with_info=True)

    ds_train = load_dataset('data/')
    ds_train_encoded = encode_examples(ds_train, tokenizer, limit=-1).shuffle(10000).batch(batch_size)
    # ds_test_encoded = encode_examples(ds_train, tokenizer, limit=100).batch(batch_size)

    learning_rate = 2e-5
    number_of_epochs = 4
    model = CZERT(args.model_dir, num_labels=578)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_train_encoded)
    predicted = predict_with_model(['I fucking hate you'], model, tokenizer)
    print(predicted)
