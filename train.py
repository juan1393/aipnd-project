import argparse
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

ACCEPTED_ARCHITECTURES = ["densenet121", "vgg13"]


def get_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader


def build_model(architecture, hidden_units):
    print("Building model...")

    if architecture not in ACCEPTED_ARCHITECTURES:
        raise Exception("Architecture not accepted")

    if architecture == ACCEPTED_ARCHITECTURES[1]:
        model = models.vgg13(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    classifier = nn.Sequential(OrderedDict(
        [
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

    model.classifier = classifier
    return model


def train_model(model, trainloader, validloader, hidden_units, arch, gpu,
                lr, epochs, save_path):
    print("Training model...")

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    if gpu and torch.cuda.is_available():
        model.cuda()

    steps = 0
    running_loss = 0
    print_every = 2

    for e in range(epochs):
        for images, labels in iter(trainloader):
            steps += 1

            # Wrap images and labels in Variables so we can calculate gradients
            if gpu and torch.cuda.is_available():
                inputs = Variable(images.cuda())
                targets = Variable(labels.cuda())
            else:
                inputs = Variable(images)
                targets = Variable(labels)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                val_loss = 0
                for ii, (images, labels) in enumerate(validloader):
                    # Set volatile to True so we don't save the history
                    if torch.cuda.is_available():
                        inputs = Variable(images.cuda(), volatile=True)
                        labels = Variable(labels.cuda(), volatile=True)
                    else:
                        inputs = Variable(images, volatile=True)
                        labels = Variable(labels, volatile=True)

                    output = model.forward(inputs)
                    val_loss += loss_fn(output, labels).data[0]

                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Validation Loss: {:.3f}.. ".format(val_loss / len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(validloader)))

                running_loss = 0

                # Make sure dropout is on for training
                model.train()

    print("Saving model...")
    save_model(model, trainloader, epochs, optimizer, save_path, lr, hidden_units, arch)


def save_model(model, trainloader, epochs, optimizer, path, lr, hidden_units, arch):
    model.class_to_idx = trainloader.dataset.class_to_idx
    model.epochs = epochs
    checkpoint = {'optimizer_dict': optimizer.state_dict(),
                  'input_size': [3, 224, 224],
                  'batch_size': trainloader.batch_size,
                  'state_dict': model.state_dict(),
                  'class_to_idx': trainloader.dataset.class_to_idx,
                  'output_size': 102,
                  'lr': lr,
                  'hidden_units': hidden_units,
                  'arch': arch,
                  'epoch': epochs}

    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Predict flower types')
    parser.add_argument('--gpu', action='store_true', help='Using GPU or not')
    parser.add_argument('data_dir', type=str, default='flowers', help='Path to images')
    parser.add_argument('--arch', type=str, default='densenet121', help='Architectures available -> densenet121, vgg13')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs')
    parser.add_argument('--save_dir', type=str, default='flower102_checkpoint.pth',
                        help='Path where model will be saved')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=500, help='Hidden units')
    args = parser.parse_args()

    trainloader, validloader, testloader = get_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    train_model(model, trainloader, validloader, args.hidden_units, args.arch,
                args.gpu, args.lr, args.epochs, args.save_dir)


if __name__ == "__main__":
    main()
