from typing import Dict, List, Optional, Any

from overrides import overrides
import torch

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.models import Model
from allennlp.models.crf_tagger import CrfTagger
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util


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
        ret = torch.cat(tensors, dim=-1)
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
                # https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py#L208  noqa
                # get_text_field_mask:
                # https://github.com/allenai/allennlp/blob/1b31320a6bb8bac6eca0ab222c45fa5db6bfe515/allennlp/nn/util.py#L411-L412  noqa
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
