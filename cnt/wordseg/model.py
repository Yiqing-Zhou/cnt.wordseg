'''
(1)
Flow:
    char_ids -> char emb -> BiLSTM -> h (timestep)

text_field_embedder.TextFieldEmbedder: {
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
        }
    }
}

seq2seq_encoder.Seq2SeqEncoder: {
    "type": "lstm",
    "bidirectional": true,
    "input_size": 100,
    "hidden_size": 100,
    "num_layers": 1,
    "dropout": 0.2
}

(2)
Flow:
    h                         --concat--> [emission rep]
    context_id -> context emb ----/
    vocab info -> (todo)      ----/

    [emission rep] -> FC -> emission prob -> CRF -> B|M|E|S

Model crf_tagger.CrfTagger.
Use a customized implementation of TextFieldEmbedder for concat op.
Use pass_through_encoder.PassThroughEncoder to skip seq2seq.
'''


