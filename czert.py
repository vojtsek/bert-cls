import tensorflow as tf
from transformers import TFBertForSequenceClassification, TFBertModel


class CZERT(tf.keras.Model):

    def __init__(self, model_dir, num_labels):
        super(CZERT, self).__init__()
        self.encoder = TFBertModel.from_pretrained(model_dir)
        self.cls_layer = tf.keras.layers.Dense(num_labels)

    def __call__(self, example, *args, **kwargs):
        output = self.encoder(example['input_ids'])
        # loss, logits , hidden_states, attentions = output[:4]
        cls_tokens = output['last_hidden_state'][:, 0, :]
        x = self.cls_layer(cls_tokens)
        return x
