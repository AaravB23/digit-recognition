# Imports PyTorch, a very powerful machine learning framework used especially for
# building and training neural networks. Works well with computer vision.
import torch

# The neural network module of PyTorch. Gives us access to tools we will use later.
import torch.nn as nn

# The optimization module of PyTorch. Gives us access to tools we will use later.
import torch.optim as optim

# TorchVision is a companion library to PyTorch, specialized towards computer vision.
# Datasets is a library of premade datasets so we don't have to find our own.
# Transforms gives us tools that help with preprocessing our images.
from torchvision import datasets, transforms

# DataLoader efficiently feeds data to the model by batching and shuffling samples.
from torch.utils.data import DataLoader

# Library that helps with plotting data / images.
import matplotlib.pyplot as plt

# To make our sample images random
from random import sample 

# For main method later, used with loading / reloading saved model.
import os

# This function will load, preprocess, and return our training and testing data.
# The paramater, batch_size defines how many samples are grouped together and
# passed through the model in each step. A 'batch' is a single unit
# of training work. That means the model preprocesses that many images
# before updating its weights after processing each batch.
def load_data(batch_size=64):

    # This preprocesses each image before it enters the model. 
    # ToTensor converts the image into a numeric format that PyTorch can work with.
    # Normalize adjusts the range so training is more consistent.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # The first parameter is the mean brightness. The second is the
        # standard deviation, showing how spread out the values are. 
        transforms.Normalize((0.5,), (0.5,))
    ])

    # This loads the MNIST training set, automatically downloading the images 
    # preprocessing using the transform object we made earlier.
    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Same as previous except for the test data.
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # DataLoader turns the dataset into batches that the model can process.
    # The train_loader has shuffle enable, so that the model sees a different
    # order per epoch. The test loader doesn't, to keep evaluation consistent.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Class that defines the structure of our neural network.
# Convolutional Neural Networks (CNN) use convolution layers to detect patterns
# in images, pooling layers to reduce image sizes and fully connected layers
# at the end to make predictions for each digit.
# The parameter, nn.Module, is the base of all PyTorch models.
class DigitCNN(nn.Module):
    # Sets up the layers of the neural network. The convolution and pooling
    # layers that extract features from the images are defined here.
    def __init__(self):
        # This initializes the nn.module(), setting up the internal workings of
        # PyTorch that tracks all the layers, weights, and gradients in the
        # network, making training work correctly.
        super(DigitCNN, self).__init__()

        # nn.Sequential() creates a callable module that contains each layer
        # in order. A layer is a block that computes a specific math operation,
        # and stacking these together creates the network.
        self.conv_layers = nn.Sequential(
            # This is a convolution layer, basically applies filters (kernels) to
            # image data to detect features. Good for finding patterns regardless
            # of their position in the image. The 1 represents the input channel,
            # as it is grayscale. The 32 means the 1 channel input gets taken and
            # creates 32 different filters, each looking for a different visual
            # pattern. These create feature maps, which is highlights specific patterns 
            # like curves, edges, etc. kernel_size=3 is how large the window that 
            # moves across the image is, so with 3 it is a 3x3 pixel window.
            nn.Conv2d(1, 32, kernel_size=3),
            # Converts all negative values to 0 and keeps positive as is. 
            # Makes it non-linear, without this step the convolutions would just be
            # one big linear operation and the network wouldn't learn anything useful.  
            nn.ReLU(),
            # Looks through the image in small blocks and keeps only the strongest
            # value from each 2x2 cluster. Made to remove small details.
            nn.MaxPool2d(2),

            # Second convolution, as the first convolution layer had 32 filters,
            # this one has those 32 as it's input. It then learns 64 filters, and
            # repeats the process from each of those 32 inputs for every new one.
            # This lets the network detect more detailed and higher-level shapes. 
            nn.Conv2d(32, 64, kernel_size=3),
            # Repeated for same reason as previous.
            nn.ReLU(),
            # Repeated to keep data manageable and to reduce noise. 
            nn.MaxPool2d(2)
        )

        # fc_layers (fully connected layers) is the part of the network that takes 
        # the compressed feature maps from the convolutional stack and turns them 
        # into digit predictions. It essentially makes the decisions after learning
        # what each digit 'looks' like.
        self.fc_layers = nn.Sequential(
            # The convolutional output has a 'shape' of (batch, 64, 5, 5) aka (how
            # many images are being processed at the same time, # of channels, height,
            # width of each feature map after shrinking), and this converts it to
            # (batch, 64 * 5 * 5), making it a vector compatible with the fc layer.
            nn.Flatten(),
            # Takes the flattened vector of size 64*5*5 and outputs a vector
            # of size 64. This combines all the extracted features into a 
            # smaller, meaningful representation that the network can work with. 
            nn.Linear(64 * 5 * 5, 64),
            # Same non-linear transformation as earlier, helps with learning more
            # complex combinations of features rather than behaving linearly. 
            nn.ReLU(),
            # Final classification layer, outputting the 10 scores. 
            nn.Linear(64, 10)
        )

    # When model(x) is called, this is what runs. 
    def forward(self, x):
        # The image tensor (multi-dimensional array that represents the image data) x
        # enters the convolutional stack, transformed into feature maps.
        x = self.conv_layers(x)
        # The flattened feature maps enter the fully connected layers, which collapse
        # the extracted features into class scores.
        x = self.fc_layers(x)
        # Returns a 10 number output tensor, with each number representing a digit.
        # These numbers are logits, which is a unnormalized score for each class
        # representing how likely the network thinks a digit class is.  
        return x

# Trains the model by repeatedly showing batches of images and adjusting the weights
# in the network. Model is the neural network, train_loader is the iterable that
# yields the batches of training data, and epochs is the number of complete passes
# through the entire training set.
def train_model(model, train_loader, epochs=5):
    # Defines the loss function that measures how far the model's predicted 
    # class scores are from the correct class.
    criterion = nn.CrossEntropyLoss()
    # Creates the algorithm that adjusts the model's weights after each batch, using
    # the model's parameters and a fixed learning rate of 0.001. Adam is a weight update
    # rule that adapts each parameter update based on past gradients and their variance.
    # A gradient is a vector that points where the greatest increase in a function is.
    # Used to optimize variables in order to minimize / maximize a function.
    # The learning rate is the scale factor for each weight update, so it controls the
    # distance the optimizer moves the parameters on every step.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Switches network into training mode.
    model.train()

    # Iterates over entire data set multiple times (epochs)
    for epoch in range(epochs):
        # Measures total batch loss per epoch. Batch loss is the error found from
        # calculating the loss function for a single batch of images. Instead of
        # calculating the total loss of the entire dataset, the model processes a
        # small group of images and measures how far its predictions are from the
        # correct labels. 
        total_loss = 0

        # Processes each batch one at a time
        for images, labels in train_loader:
            # Clears the gradients stored in the model's parameters, as PyTorch
            # accumulates gradients. Without calling this, they would keep stacking
            # and prevent proper training. 
            optimizer.zero_grad()

            # Feeds the batch of input images into the model, calling forward().
            # Processes the images through layers, returning the tensor of logits.
            outputs = model(images)
            # Finds loss of the predictions compared to real labels.
            loss = criterion(outputs, labels)

            # Computes the gradient of loss considering every parameter in the
            # network, calculating the weight contributed to the error so the
            # optimizer knows how to adjust them. The gradient is a measure
            # of how much a parameter in the network is affecting the loss.
            # It tells the optimizer which direction to move each weight.
            loss.backward()
            # Updates the model parameters using gradients calculated in the
            # previous line. Applies the Adam rule we set earlier. 
            optimizer.step()

            # Adds to the cumulative loss.
            total_loss += loss.item()
        # Shows total loss after whole epoch (whole data set). 
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

# Testing model accuracy.
def evaluate_model(model, test_loader):
    # Switches the model into evaluation mode.
    model.eval()
    # Counts # of images predicted correctly.
    correct = 0
    # Counts # of images processed from test set.
    total = 0

    # Creates environment where gradient tracking is disabled, saving memory and speed.
    with torch.no_grad():
        # Loops over test dataset in batches like the training.
        for images, labels in test_loader:
            # Runs forward like the training, returning tensors of logits.
            outputs = model(images)
            # Finds the predicted class for each image in the batch. outputs.data
            # contains the logits from the model. torch.max return which digit has
            # the highest score. The index corresponds with the digit (0-9).
            # _ ignores the actual max value, as we just care about the digit.
            _, predicted = torch.max(outputs.data, 1)

            # Adds however many images were in the batch. The 0 represents the 1st
            # dimension of the tensor, 
            total += labels.size(0)
            # If the predicted result is the same as the actual label value, adds
            # to the correct counter. sum() adds up the True values from the batch.
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Displays a few sample images from the test set, along with the predicted values.
def show_predictions(model, test_loader):
    model.eval()
    dataset = test_loader.dataset
    # Picks 5 indices randomly from the dataset
    indices = sample(range(len(dataset)), 5)

    for i in indices:
        image, _ = dataset[i]
        output = model(image.unsqueeze(0))
        _, pred = torch.max(output, 1)

        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Prediction: {pred.item()}")
        plt.axis("off")
        plt.show()

def main():
    # Where the model will be saved
    model_path = "digit_cnn.pth"

    print("Loading data...")
    train_loader, test_loader = load_data()

    # Initialize the model with the type of algorithm we will use.
    print("Building model...")
    model = DigitCNN()

    # Ask user if they want to use a saved model.
    use_saved = input("Load saved model? (y/n): ").strip().lower()

    if use_saved == "y":
        if os.path.exists(model_path):
            # Loads network exactly how it was with the saved model weights.
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Loaded saved model.")
            print("Evaluating...")
            evaluate_model(model, test_loader)
            print("Showing sample predictions...")
            show_predictions(model, test_loader)
            # End execution
            return
        else:
            print("No saved model found, training new model...")

    print("Training...")
    train_model(model, train_loader, epochs=5)

    # Save the learned weights to be reloaded later.
    torch.save(model.state_dict(), model_path)

    print("Evaluating...")
    evaluate_model(model, test_loader)

    print("Showing sample predictions...")
    show_predictions(model, test_loader)

if __name__ == "__main__":
    main()