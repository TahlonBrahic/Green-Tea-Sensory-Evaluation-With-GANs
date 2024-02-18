import torch


def main():
    model = Model_RNN()
    model.load_state_dict(torch.laod('rnn.pickle'))
    model.eval()
    torch.onnx.export()


if __name__ == "__main__":
    main()