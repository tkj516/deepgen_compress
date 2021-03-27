import torch
from torchsummary import summary
from vq_tests.edsr_arch import EDSR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Initialize the EDSR network
    model = EDSR(num_in_ch=1, num_out_ch=1).to(device)

    print(model)

    summary(model, (1, 64, 64))