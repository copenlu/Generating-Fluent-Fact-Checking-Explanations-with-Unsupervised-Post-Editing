import torch


class ClassifierModel(torch.nn.Module):
    """Prediction is conditioned on the extracted sentences."""
    def __init__(self, args, transformer_model, transformer_config):
        super().__init__()
        self.args = args
        self.transformer_model = transformer_model

        # pooler
        self.dense = torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size)
        self.activation_tanh = torch.nn.Tanh()

        # classification
        self.dropout = torch.nn.Dropout(transformer_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(transformer_config.hidden_size, transformer_config.num_labels)

        self.softmax_pred = torch.nn.Softmax(dim=-1)

    def encode(self, token_ids):
        if len(token_ids.size()) > 2:
            return \
            self.transformer_model(inputs_embeds=token_ids)[0]
        else:
            return \
            self.transformer_model(token_ids, attention_mask=token_ids != 0)[0]
            # attention_mask=attention_mask,

    def forward(self, token_ids=None, batch=None):
        if token_ids == None:
            token_ids = batch['input_ids_tensor']
        hidden_states = self.encode(token_ids)

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_tanh(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits_pred = self.classifier(pooled_output)

        return logits_pred


class ClassifyExtractJointModel(torch.nn.Module):
    """
    Classify the instances and use the predicted logits as input for the
    extraction component.
    """
    def __init__(self, args, transformer_model,
                 transformer_config,):
        super().__init__()
        self.args = args
        self.transformer_model = transformer_model

        # pooler
        self.dense = torch.nn.Linear(transformer_config.hidden_size,
                                     transformer_config.hidden_size)
        self.activation_tanh = torch.nn.Tanh()

        # classification
        self.dropout = torch.nn.Dropout(transformer_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(transformer_config.hidden_size,
                                          transformer_config.num_labels)

        # pooler sentences
        self.dense_sentences = torch.nn.Linear(
            transformer_config.hidden_size+args.labels,
            transformer_config.hidden_size+args.labels)
        self.classifier_sentences = torch.nn.Linear(transformer_config.hidden_size+args.labels, args.labels)

    def forward(self, token_ids=None, batch=None):

        if token_ids == None:
            token_ids = batch['input_ids_tensor']

        attention_mask = token_ids != 0

        if len(token_ids.size()) > 2:
            outputs = self.transformer_model(
                inputs_embeds=token_ids,
                # attention_mask=attention_mask,
            )
        else:
            outputs = self.transformer_model(
                token_ids,
                attention_mask=attention_mask,
            )

        hidden_states = outputs[0]

        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits_pred = self.classifier(pooled_output)

        result = [logits_pred]

        logits_pred_extended = logits_pred.unsqueeze(1).repeat(1, batch['sentences_mask'].size(1), 1)

        cls_tokens_tensor = hidden_states[
            torch.arange(hidden_states.size(0)).unsqueeze(1), batch[
                'sentences_idx_tensor']]
        cls_tokens_tensor = cls_tokens_tensor * batch[
            'sentences_mask'].unsqueeze(2) + \
                            (~batch['sentences_mask'].unsqueeze(2)) * -1.0e3
        cls_tokens_tensor = torch.cat([cls_tokens_tensor, logits_pred_extended], dim=-1)

        pooled_output_sentences = self.dense_sentences(cls_tokens_tensor)
        pooled_output_sentences = self.activation_tanh(pooled_output_sentences)
        pooled_output_sentences = self.dropout(pooled_output_sentences)
        sentences_pred = self.classifier_sentences(pooled_output_sentences)
        result.append(sentences_pred)

        pred_label = torch.max(logits_pred, dim=-1)[1]
        pred_label_extended = pred_label.unsqueeze(1).repeat(1, batch[
            'sentences_mask'].size(1)).view(-1, 1)
        logits_sentences_imp = sentences_pred.view(-1, self.args.labels)[
            torch.arange(
                sentences_pred.view(-1, self.args.labels).size(0)).unsqueeze(
                1),
            pred_label_extended
        ].view(batch['sentences_mask'].size(0), batch['sentences_mask'].size(1))
        result.append(logits_sentences_imp)

        return result


class ExtractClassifyJointModel(torch.nn.Module):
    """Prediction is conditioned on the extracted sentences."""
    def __init__(self, args, transformer_model, transformer_config):
        super().__init__()
        self.args = args
        self.transformer_model = transformer_model

        # pooler
        self.dense = torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size)
        self.activation_tanh = torch.nn.Tanh()

        # classification
        self.dropout = torch.nn.Dropout(transformer_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(transformer_config.hidden_size, transformer_config.num_labels)

        # pooler sentences
        self.dense_sentences = torch.nn.Linear(transformer_config.hidden_size,
                                     transformer_config.hidden_size)
        # classification
        self.classifier_sentences = torch.nn.Linear(transformer_config.hidden_size, transformer_config.num_labels)

        self.softmax_pred = torch.nn.Softmax(dim=-1)
        self.softmax_sent = torch.nn.Softmax(dim=1)

    def encode(self, token_ids, attention_mask):
        if len(token_ids.size()) > 2:
            encoded_batch = self.transformer_model(
                inputs_embeds=token_ids,
                # attention_mask=attention_mask,
            )[0]
        else:
            encoded_batch = self.transformer_model(
                token_ids,
                attention_mask=attention_mask,
            )[0]

        return encoded_batch

    def forward(self, token_ids=None, batch=None):
        if token_ids == None:
            token_ids = batch['input_ids_tensor']

        attention_mask = token_ids != 0

        hidden_states = self.encode(token_ids, attention_mask)


        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_tanh(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits_pred = self.classifier(pooled_output)

        cls_tokens_tensor = hidden_states[
            torch.arange(hidden_states.size(0)).unsqueeze(1), batch[
                'sentences_idx_tensor']]
        cls_tokens_tensor = cls_tokens_tensor * batch[
            'sentences_mask'].unsqueeze(2) + \
                            (~batch['sentences_mask'].unsqueeze(2)) * -1.0e3

        pooled_output_sentences = self.dense_sentences(cls_tokens_tensor)
        pooled_output_sentences = self.activation_tanh(pooled_output_sentences)
        pooled_output_sentences = self.dropout(pooled_output_sentences)

        logits_sentences = self.classifier_sentences(pooled_output_sentences)

        final_logits_pred = torch.einsum('bij,bj->bj',
                                         self.softmax_sent(logits_sentences),
                                         self.softmax_pred(logits_pred))
        result = [final_logits_pred]
        result.append(logits_sentences.squeeze(-1))

        pred_label = torch.max(final_logits_pred, dim=-1)[1]
        pred_label_extended = pred_label.unsqueeze(1).\
            repeat(1, batch['sentences_mask'].size(1)).view(-1, 1)
        logits_sentences_imp = logits_sentences.view(-1, self.args.labels)[
            torch.arange(logits_sentences.
                         view(-1, self.args.labels).size(0)).unsqueeze(1),
            pred_label_extended
        ].view(batch['sentences_mask'].size(0), batch['sentences_mask'].size(1))
        result.append(logits_sentences_imp)

        return result