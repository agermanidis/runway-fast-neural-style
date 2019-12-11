import runway
import torch
from torchvision import transforms
from transformer_net import TransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@runway.setup(options={'checkpoint': runway.file(extension='.pth')})
def setup(opts):
    style_model = TransformerNet()
    state_dict = torch.load(opts['checkpoint'])
    style_model.load_state_dict(state_dict)
    style_model.to(device)
    return style_model    


command_inputs = {
    'image': runway.image
}

command_outputs = {
    'output': runway.image
}

@runway.command('stylize', inputs=command_inputs, outputs=command_outputs)
def stylize(model, inputs):
    content_img = inputs['image']
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_img = content_transform(content_img)
    content_img = content_img.unsqueeze(0).to(device)
    data = model(content_img).detach().cpu()
    img = data[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img


if __name__ == '__main__':
    runway.run()
