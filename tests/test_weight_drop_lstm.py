import torch
from cnt.wordseg.weight_drop_lstm import WeightDropoutLSTM


def test_weight_dropout_lstm():
    lstm = WeightDropoutLSTM(
        input_size=10,
        hidden_size=5,
        num_layers=4,
        weight_dropout=0.9,
    )
    lstm.cuda()
    tensor_in = torch.randn(2, 1, 10, requires_grad=True).cuda()

    run1 = [x.sum() for x in lstm(tensor_in)[0].data]
    run2 = [x.sum() for x in lstm(tensor_in)[0].data]

    assert run1[0] == run2[0]
    assert run1[1] != run2[1]


def test_weight_dropout_lstm_bidirectional():
    lstm = WeightDropoutLSTM(
        input_size=10,
        hidden_size=5,
        num_layers=4,
        bidirectional=True,
        weight_dropout=0.9,
    )
    lstm.cuda()
    tensor_in = torch.randn(2, 1, 10, requires_grad=True).cuda()

    run1 = [x.sum() for x in lstm(tensor_in)[0].data]
    run2 = [x.sum() for x in lstm(tensor_in)[0].data]

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]
