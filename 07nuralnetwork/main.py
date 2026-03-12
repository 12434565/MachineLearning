
import torch ## torch let's us create tensors and also provides helper functions
import torch.nn as nn ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.nn.functional as F # This gives us relu()
from torch.optim import SGD # SGD is short of Stochastic Gradient Descent, but
                            # the way we'll use it, passing in all of the training
                            # data at once instead of passing it random subsets,
                            # it will act just like plain old Gradient Descent.

import lightning as L ## Lightning makes it easier to write, optimize and scale our code
from torch.utils.data import TensorDataset, DataLoader ## We'll store our data in DataLoaders


import matplotlib
matplotlib.use("Agg")   # 必须在 pyplot 之前
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import roc_curve, auc


## NOTE: If you get an error running this block of code, it is probably
##       because you installed a new package earlier and forgot to
##       restart your session for python to find the new module(s).
##
##       To restart your session:
##       - In Google Colab, click on the "Runtime" menu and select
##         "Restart Session" from the pulldown menu
##       - In a local jupyter notebook, click on the "Kernel" menu and select
##         "Restart Kernel" from the pulldown menu

"""----

# Create the Training Dataset

In Chapter 2, we had a very simple dataset that consisted of three points, as seen in the figure below.

<img src="https://github.com/StatQuest/signa/blob/main/chapter_02/images/chapter_2_training_data.png?raw=1" alt="a simple dataset for training" style="width: 800px;">

Although it's not required, we're going to put our training data into a `DataLoader`. `DataLoaders` are easy to make and they offer a lot of cool features. For example, if we had a large dataset, a `DataLoader` gives us a super easy way to access the data in batches instead of all at once. This is critical when we have more data than RAM to store it in. A DataLoader can also shuffle the data for us each epoch and makes it easy to only use a fraction of the data if we want to do a quick and rough training for debugging purposes.
"""

## The inputs are the x-axis coordinates for each data point
## These values represent different doses

import pandas as pd
import numpy as np

train_df = pd.read_csv("06_bio_single_train.csv")

training_inputs = torch.tensor(
    train_df["cpg_density"].values,
    dtype=torch.float32
)

training_labels = torch.tensor(
    train_df["expressed"].values,
    dtype=torch.float32
)


## Now let's package everything up into a DataLoader...
training_dataset = TensorDataset(training_inputs, training_labels)

dataloader = DataLoader(
    training_dataset
)


"""----

# Create a Neural Network with Trainable Weights and Biases
<a id="create"></a>

Now we'll build a neural network that has trainable Weights and Biases. For this, we'll use `L.LightningModule`, which has everything `nn.Module` has, plus we can define the optimizer we want to use as well as tell PyTorch how each training step should work.
"""

class myNN(L.LightningModule):

    def __init__(self):

        super().__init__()

        ## Create all of the weights and biases for the network.
        ## However, this time they are initialized with random values.
        ## We are also wrapping the tensors up in nn.Parameter() objects.
        ## PyTorch will only optimize parameters. There are a lot of
        ## different ways to create parameters, and we'll see those
        ## in later examples, but nn.Parameter() is the most basic.
        self.w1 = nn.Parameter(torch.tensor(0.06))
        self.b1 = nn.Parameter(torch.tensor(0.0))

        self.w2 = nn.Parameter(torch.tensor(3.49))
        self.b2 = nn.Parameter(torch.tensor(0.0))

        self.w3 = nn.Parameter(torch.tensor(-4.11))
        self.w4 = nn.Parameter(torch.tensor(2.74))

        self.loss = nn.MSELoss(reduction='sum')


    def forward(self, input_values):
        ## The forward() method is identical to what we used in Chapter 1.

        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values


    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return SGD(self.parameters(), lr=0.01)
        ## NOTE: PyTorch doesn't have a Gradient Descent optimizer, just a
        ## Stochastic Gradient Descent (SGD) optimizer. However, since we
        ## are running all 3 doses through the NN each time, rather than a
        ## random subset, we are essentially doing Gradient Descent instead of
        ## SGD.


    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        ## NOTE: When training_step() is called it calculates the loss with the code below...
        inputs, labels = batch # collect input
        outputs = self.forward(inputs) # run input through the neural network
        loss = self.loss(outputs, labels) ## the `loss` quantifies the difference between
                                          ## the observed drug effectiveness in `labels`
                                          ## and the outputs created by the neural network

        return loss


model = myNN() # First, make model from the class

## Now print out the name and value for each named parameter
## parameter in the model. Remember parameters are variables,
## like Weights and Biases, that we can train.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))

## now run different doses through the neural network.
output_values = model(training_inputs)
torch.round(output_values, decimals=2)

"""# BAM!

We successfully ran the doses from the training data through the model. However, the output from the model is way different than we expect (we expected 0.0, 1.0, and 0.0). So let's draw a picture of the bent shape that the model uses to make predictions and compare that to the training data.
"""

## Create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)

# now print out the doses to make sure they are what we expect...
input_doses

output_values = model(input_doses)
output_values

## Now draw a graph that shows how well, or poorly, the model
## predicts the training data. At this point, since the
## model is untrained, there should be a big difference between
## the model's output and the training data.

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## First, draw the individual output points
sns.scatterplot(x=input_doses,
                y=output_values.detach().numpy(),
                color='green',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=output_values.detach().numpy(), ## NOTE: We call .detatch() because...
             color='green',
             linewidth=2.5)

## Add the values in the training dataset
sns.scatterplot(x=training_inputs,
                y=training_labels,
                color='orange',
                s=200)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

"""# DOUBLE BAM!!

Now that we see how badly the bent shape fits the training data, let's train the model.

-----

# Training the Weights and Biases in the Neural Network

Training consists of creating a **Lightning Trainer** with `L.Trainer()` and then calling the `fit()` method on the our model with the training data.
"""

model = myNN()
## Now train the model...
trainer = L.Trainer(max_epochs=500, # how many times to go through the training data
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False)

trainer.fit(model, train_dataloaders=dataloader)
# ===== ROC on test set =====
test_df = pd.read_csv("06_bio_single_test.csv")
test_inputs = torch.tensor(test_df["cpg_density"].values, dtype=torch.float32)
test_labels = torch.tensor(test_df["expressed"].values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    logits = model(test_inputs)
    probs = torch.sigmoid(logits)

fpr, tpr, _ = roc_curve(test_labels.numpy(), probs.numpy())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.savefig("roc.png", dpi=200, bbox_inches="tight")
plt.close()
## Now that we've trained the model, let's print out the
## new values for each Weight and Bias.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=3))

"""Lastly, let's draw a graph of the bent shape that the model is using for predictions and compare it to the training data. In theory, the bent shape should fit the data much better now that we have optimized the Weights and Biases."""

## now run the different doses through the neural network.
output_values = model(input_doses)
torch.round(output_values, decimals=2)

## Now draw a graph that shows how well, or poorly, the model
## predicts the training data. At this point, since we just
## trained th model, the training data should overlap the
## model's output

## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## First, draw the individual output points
sns.scatterplot(x=input_doses,
                y=output_values.detach().numpy(),
                color='green',
                s=200)

## Now connect those points with a line
sns.lineplot(x=input_doses,
             y=output_values.detach().numpy(), ## NOTE: We call .detatch() because...
             color='green',
             linewidth=2.5)

## Add the values in the training dataset
sns.scatterplot(x=training_inputs,
                y=training_labels,
                color='orange',
                s=200)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.savefig("plot.png", dpi=200, bbox_inches="tight")
plt.close()



