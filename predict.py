import argparse
import json
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from collections import OrderedDict
from PIL import Image
import numpy as np


def load_checkpoint(path):
    checkpoint = torch.load(path)

    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']

    if arch == "vgg13":
        model = models.vgg13()
    else:
        model = models.densenet121()

    classifier = nn.Sequential(OrderedDict(
        [
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {i: k for k, i in class_to_idx.items()}
    return model, class_to_idx, idx_to_class


def process_image(image):
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0] // 2 - (224 / 2),
        size[1] // 2 - (224 / 2),
        size[0] // 2 + (224 / 2),
        size[1] // 2 + (224 / 2))
    )
    np_image = np.array(image) / 255
    np_image[:, :, 0] = (np_image[:, :, 0] - 0.485) / (0.229)
    np_image[:, :, 1] = (np_image[:, :, 1] - 0.456) / (0.224)
    np_image[:, :, 2] = (np_image[:, :, 2] - 0.406) / (0.225)
    np_image = np.transpose(np_image, (2, 0, 1))

    return np_image


def predict(image_path, model, idx_to_class, cat_to_name, topk, gpu):
    if gpu and torch.cuda.is_available():
        model.cuda()

    image = process_image(Image.open(image_path))
    image = torch.FloatTensor([image])
    model.eval()

    if gpu and torch.cuda.is_available():
        inputs = Variable(image.cuda())
    else:
        inputs = Variable(image)

    output = model.forward(inputs)
    ps = torch.exp(output).data.cpu().numpy()[0]

    topk_index = np.argsort(ps)[-topk:][::-1]
    topk_class = [idx_to_class[x] for x in topk_index]
    named_topk_class = [cat_to_name[x] for x in topk_class]
    topk_prob = ps[topk_index]

    return topk_prob, named_topk_class


def label_mapping(cat_to_name):
    with open(cat_to_name, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Predict flower types')
    parser.add_argument('--gpu', action='store_true', help='Using GPU or not')
    parser.add_argument('input', type=str, help='Path to image that will be predicted')
    parser.add_argument('checkpoint', type=str, help='Path to training checkpoint')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to category to flower name mapping json')
    parser.add_argument('--topk', type=int, default=5, help='Top k probabilities')

    args = parser.parse_args()

    cat_to_name = label_mapping(args.category_names)

    model, class_to_idx, idx_to_class = load_checkpoint(args.checkpoint)
    topk_prob, named_topk_class = predict(args.input, model, idx_to_class,
                                          cat_to_name, args.topk, args.gpu)

    print(dict(zip(named_topk_class, topk_prob)))


if __name__ == "__main__":
    main()
