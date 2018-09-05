from typing import Dict, List, Optional, Any

from overrides import overrides

import torch
import torch.nn as nn

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import SpanBasedF1Measure

from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.nn import Activation


@TextFieldEmbedder.register("crf_concat_embedder")
class CrfConcatEmbedder(TextFieldEmbedder):

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self._output_dim = output_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,  # pylint: disable=arguments-differ
        text_field_input: Dict[str, torch.Tensor],
        num_wrapping_dims: int = 0,
    ) -> torch.Tensor:
        """
        text_field_input: value shape = (batch_size, num_tokens, X_i)
        out: (batch_size, num_tokens, for all i concat(X_i))
        """
        # error will be raised if shapes are not matched.
        tensors = tuple(
            t
            for name, t in text_field_input.items() if name != 'mask'
        )
        if len(tensors) > 1:
            ret = torch.cat(tensors, dim=-1)
        else:
            ret = tensors[0]
        assert ret.shape[-1] == self._output_dim
        return ret


@Model.register("cnt_wordseg")
class CntWordSeg(Model):
    """
    Multi-criteria CWS.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        tokens_embedder: TextFieldEmbedder,
        context_embedder: TextFieldEmbedder,
        tokens_seq2seq: Seq2SeqEncoder,
        tokens_context_crf: Model,
        dropout: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:

        super().__init__(vocab, regularizer)

        # shared dropout.
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        # (1)
        self._tokens_embedder = tokens_embedder
        self._tokens_seq2seq = tokens_seq2seq

        # (2)
        self._context_embedder = context_embedder
        self._crf_tagger = tokens_context_crf

        initializer(self)

    # copy from CrfTagger.
    def forward(
        self,
        # (batch_size, num_tokens)
        tokens: Dict[str, torch.LongTensor],
        # (batch_size, 1)
        context: Dict[str, torch.LongTensor],
        # (batch_size, num_tokens)
        tags: torch.LongTensor = None,
        # (batch_size, num_tokens)
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # (1)
        embedded_tokens = self._tokens_embedder(tokens)
        embedded_context = self._context_embedder(context)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_tokens = self.dropout(embedded_tokens)
            embedded_context = self.dropout(embedded_context)

        encoded_tokens = self._tokens_seq2seq(embedded_tokens, mask)

        # (2)
        # (batch_size, 1, context_dim) -> (batch_size, num_tokens, context_dim)
        embedded_context = embedded_context.expand(
            -1, encoded_tokens.shape[1], -1,
        )
        assert embedded_context.shape[:-1] == encoded_tokens.shape[:-1]

        return self._crf_tagger(
            tokens={
                'tokens': encoded_tokens,
                'context': embedded_context,
                # pass mask to crf_tagger.
                # CrfTagger.forward.
                # https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L208
                # get_text_field_mask:
                # https://github.com/allenai/allennlp/blob/1b31320a6bb8bac6eca0ab222c45fa5db6bfe515/allennlp/nn/util.py#L411-L412
                'mask': mask,
            },
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa
        return self._crf_tagger.decode(output_dict=output_dict)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._crf_tagger.get_metrics(reset=reset)


@Model.register("cnt_wordseg_inject_tag")
class CntWordSegInjectTag(Model):
    """
    Multi-criteria CWS.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        # emission.
        tokens_embedder: TextFieldEmbedder,
        tokens_seq2seq: Seq2SeqEncoder,
        # tokens_bigram_embedder: TextFieldEmbedder,
        # crf.
        tokens_context_crf: Model,
        # configs.
        dropout: Optional[float] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:

        super().__init__(vocab, regularizer)

        # for generating emission prob.
        self._tokens_embedder = tokens_embedder
        self._tokens_seq2seq = tokens_seq2seq
        # self._tokens_bigram_embedder = tokens_bigram_embedder

        initializer(self)

        self._crf_tagger = tokens_context_crf
        # hack to ignore S-IGN.
        # self._crf_tagger._f1_metric = SpanBasedF1Measure(
        #     vocab,
        #     tag_namespace=self._crf_tagger.label_namespace,
        #     label_encoding=self._crf_tagger.label_encoding,
        #     ignore_classes=['IGN'],
        # )

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    # copy from CrfTagger.
    def forward(
        self,
        # (batch_size, num_tokens)
        tokens: Dict[str, torch.LongTensor],
        # (batch_size, num_tokens + 1)
        # tokens_bigram: Dict[str, torch.LongTensor],
        # (batch_size, num_tokens)
        tags: torch.LongTensor = None,
        # (batch_size, num_tokens)
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(tokens)
        batch_size, seqlen = mask.shape

        # (batch_size, num_tokens, X)
        embedded_tokens = self._tokens_embedder(tokens)
        encoded_tokens = self._tokens_seq2seq(embedded_tokens, mask)

        context_removed_mask = mask.clone()

        mask_offset = torch.sum(mask, dim=-1, dtype=torch.uint8) - 1
        mask_offset.unsqueeze_(dim=1)

        mask_indices = torch.arange(start=0, end=seqlen, dtype=torch.uint8, device=mask_offset.device, requires_grad=False)
        mask_indices.expand(batch_size, -1)

        context_removed_mask[mask_indices >= mask_offset] = 0

        # remove begin tag.
        embedded_tokens = embedded_tokens.narrow(1, 1, seqlen - 1).contiguous()
        encoded_tokens = encoded_tokens.narrow(1, 1, seqlen - 1).contiguous()
        context_removed_mask = context_removed_mask.narrow(1, 1, seqlen - 1).contiguous()
        if tags is not None:
            tags = tags.narrow(1, 1, seqlen - 1).contiguous()

        # embedded_tokens_bigram = self._tokens_bigram_embedder(tokens_bigram)

        # left_indices = torch.ones(batch_size, seqlen + 1, dtype=torch.uint8)
        # left_indices[:, -1] = 0
        # left_embedded_tokens_bigram = \
        #     embedded_tokens_bigram[left_indices].view(batch_size, seqlen, -1)

        # right_indices = torch.ones(batch_size, seqlen + 1, dtype=torch.uint8)
        # right_indices[:, 0] = 0
        # right_embedded_tokens_bigram = \
        #     embedded_tokens_bigram[right_indices].view(batch_size, seqlen, -1)

        # emission_rep = torch.cat(
        #     [left_embedded_tokens_bigram, encoded_tokens, right_embedded_tokens_bigram],
        #     dim=-1,
        # )

        return self._crf_tagger(
            tokens={
                'embedded_tokens': embedded_tokens,
                'encoded_tokens': encoded_tokens,
                'mask': context_removed_mask,
            },
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa
        return self._crf_tagger.decode(output_dict=output_dict)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._crf_tagger.get_metrics(reset=reset)


@Model.register("cnt_wordseg_benchmark_google")
class CntWordSegBenchmarkGoogle(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        # char emb.
        tokens_embedder: TextFieldEmbedder,
        tokens_bigram_embedder: TextFieldEmbedder,
        # lstm config.
        stacked_bilstm_hidden_size: int = 256,
        stacked_bilstm_dropout: float = 0.2,
        # projection.
        projection_dropout: float = 0.2,
        projection_activation: Optional[Activation] = None,
        # misc.
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:

        super().__init__(vocab, regularizer)

        # char emb.
        self.tokens_embedder = tokens_embedder
        self.tokens_bigram_embedder = tokens_bigram_embedder

        # stacked bilstm.
        self.stacked_bilstm_input_size = (
            tokens_embedder.get_output_dim()
            + tokens_bigram_embedder.get_output_dim() * 2
        )
        self.stacked_bilstm_hidden_size = stacked_bilstm_hidden_size
        self.stacked_bilstm_dropout = stacked_bilstm_dropout

        self.lstm_fwd = AugmentedLstm(
            self.stacked_bilstm_input_size, self.stacked_bilstm_hidden_size,
            go_forward=True,
            recurrent_dropout_probability=self.stacked_bilstm_dropout,
            use_input_projection_bias=False,
        )
        self.lstm_fwd = PytorchSeq2SeqWrapper(self.lstm_fwd)

        self.lstm_bwd = AugmentedLstm(
            # use the hidden state from `lstm_fwd`.
            self.stacked_bilstm_hidden_size, self.stacked_bilstm_hidden_size,
            go_forward=False,
            recurrent_dropout_probability=self.stacked_bilstm_dropout,
            use_input_projection_bias=False,
        )
        self.lstm_bwd = PytorchSeq2SeqWrapper(self.lstm_bwd)

        # output tagging.
        self.bmes_label_namespace = 'labels'
        self.bmes_projection_dropout = projection_dropout

        if projection_activation is None:
            self.bmes_projection_activation = lambda x: x
        else:
            self.bmes_projection_activation = projection_activation

        self.bmes_projection = FeedForward(
            input_dim=self.stacked_bilstm_hidden_size,
            num_layers=1,
            hidden_dims=4,
            activations=self.bmes_projection_activation,
            dropout=self.bmes_projection_dropout,
        )

        # loss.
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

        # metrics.
        self.metric_span = SpanBasedF1Measure(
            vocab,
            tag_namespace=self.bmes_label_namespace,
            label_encoding='BMES',
        )
        self.metric_acc = CategoricalAccuracy()

        initializer(self)

    def forward(
        self,
        # chars: unigram, bigram.
        tokens: Dict[str, torch.LongTensor],
        tokens_bigram_fwd: Dict[str, torch.LongTensor],
        tokens_bigram_bwd: Dict[str, torch.LongTensor],
        # label & misc.
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(tokens)
        batch_size, seqlen = mask.shape

        # char emb.
        emb_uni = self.tokens_embedder(tokens)
        emb_bi_fwd = self.tokens_bigram_embedder(tokens_bigram_fwd)
        emb_bi_bwd = self.tokens_bigram_embedder(tokens_bigram_bwd)

        emb_concat = torch.cat([emb_bi_fwd, emb_uni, emb_bi_bwd], dim=-1)

        # stacked lstm.
        fwd_output = self.lstm_fwd(emb_concat, mask)
        bwd_output = self.lstm_bwd(fwd_output, mask)

        # projection.
        bmes_prob = self.bmes_projection(bwd_output)
        bmes_tags = torch.argmax(bmes_prob, dim=-1)

        output = {
            'mask': mask,
            'logits': bmes_prob,
            'tags': bmes_tags,
        }

        if tags is not None:
            # copy & masking.
            flatten_tags = tags.clone()
            flatten_tags[mask == 0] = -100
            flatten_tags = flatten_tags.view(-1)

            # reshapre and calculate loss.
            flatten_bmes_prob = bmes_prob.view(batch_size * seqlen, -1)
            output['loss'] = self.loss(flatten_bmes_prob, flatten_tags)

            class_probabilities = bmes_prob * 0.
            for i, instance_tags in enumerate(bmes_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            # label acc.
            self.metric_acc(class_probabilities, tags, mask.float())
            # span f1.
            self.metric_span(class_probabilities, tags, mask.float())

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # noqa
        output_dict['tags'] = [
            [
                self.vocab.get_token_from_index(tag, namespace=self.bmes_label_namespace)
                for tag in instance_tags
            ]
            for instance_tags in output_dict['tags']
        ]

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ret = {
            x: y
            for x, y in self.metric_span.get_metric(reset=reset).items()
            if "overall" in x
        }
        ret['accuracy'] = self.metric_acc.get_metric(reset=reset)
        return ret
