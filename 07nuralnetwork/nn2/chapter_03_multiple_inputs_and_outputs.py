# -*- coding: utf-8 -*-
import torch # torch will allow us to create tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam
import lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data
import pandas as pd # We'll use pandas to read in the data and normalize it
from sklearn.model_selection import train_test_split # We'll use this to create training and testing datasets
import matplotlib
matplotlib.use("Agg")   # 必须在 pyplot 之前
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
## We'll read in the dataset with the pandas function read_table()
## read_table() can read in various text files including, comma-separated and tab-delimted.
url = "./07_bio_multi_train.csv"
df = pd.read_table(url, sep=",", header=0)
## NOTE: If the data were tab-delimted, we would set sep="\t".
## print out the first handful of rows using the head() method
## To name each column, we assign a list of column names to `columns`

## To verify we did that correctly, let's print out the first few rows
print(df.head())
df.shape ## shape returns the rows and colunns...
## To determine the number of iris species in the dataset,
## we'll count the number of unique values in the column called `class`.
df['expressed'].nunique()
## We can print out the unique values in a dataframe's column with the 'unique()' method.
df['expressed'].unique()

for class_name in df['expressed'].unique(): # for each unique class name...
    ## ...print out the number of rows associated with it
    print(class_name, ": ", sum(df['expressed'] == class_name), sep="")

## Print out the first few rows of just the `petal_width` and `sepal_width` columns
df[['cpg_density',
   'tata_box_score',
   'promoter_gc_frac',
   'h3k4me3_signal',
   'atac_accessibility',
   'tf_motif_count']].head()
input_values = df[['cpg_density',
                   'tata_box_score',
                   'promoter_gc_frac',
                   'h3k4me3_signal',
                   'atac_accessibility',
                   'tf_motif_count']]
input_values.head()
label_values = df['expressed']
label_values.head()
## Convert the strings in the 'class' column into numbers with factorize()
classes_as_numbers = label_values.factorize()[0] ## NOTE: factorize() returns a list of lists,
                                                 ## and since we only need the first list of values,
                                                 ## we index the output of factorize() with [0].
classes_as_numbers ## print out the numbers

input_train, input_test, label_train, label_test = train_test_split(input_values,
                                                                    classes_as_numbers,
                                                                    test_size=0.25,
                                                                    stratify=classes_as_numbers,
                                                                    random_state=42)
df2 = pd.read_csv("./07_bio_multi_test.csv", header=0)
df2.head()
input_values2 = df2[['cpg_density',
                   'tata_box_score',
                   'promoter_gc_frac',
                   'h3k4me3_signal',
                   'atac_accessibility',
                   'tf_motif_count']]
input_values2.head()
label_values2 = df2[["expressed"]]
label_values2.head()
input_train = input_values
input_test = input_values2
label_train = label_values
label_test = label_values2
input_train.shape
label_train.shape
input_test.shape
label_test.shape

## Now create a new tensor with one-hot encoded rows for each row in the original dataset.
one_hot_label_train = F.one_hot(torch.tensor(label_train)).type(torch.float32)
## Print out a few of the rows one-hot encoded data.
one_hot_label_train[:10]
"""So, as we can see in the output above, `classes_as_numbers` was correctly one-hot encoded and saved in `one_hot_label_train`.
Now, let's normalize the input variables so that their values range from 0 to 1. Normalizing data, so that it's all on the same scale, often makes it easier to train machine learning methods. In this case, since we have two datasets, `input_train` and `input_test`, we'll start determining the maximum and minimum values in `input_train`. Then we will use those values to normalize `input_train` and `input_test`. Using the maximum and minimum values from `input_train` to normalize both datasets avoids something called **Data Leakage**.
**NOTE:** If you don't know what it means to **normalize** your data, check out this **[short song](https://youtube.com/shorts/oZ9SrkF_-LE?feature=share)** that has a good beat, and you can dance to it.
"""
## First, determine the maximum values in input_train...
max_vals_in_input_train = input_train.max()
## Now print them out...
max_vals_in_input_train
## Second, determine the minimum values in input_train
min_vals_in_input_train = input_train.min()
## Now print them out...
min_vals_in_input_train
## Now normalize input_train with the maximum and minimum values from input_train
input_train = (input_train - min_vals_in_input_train) / (max_vals_in_input_train - min_vals_in_input_train)
input_train.head()
## Now normalize input_test with the maximum and minimum values from input_train
input_test = (input_test - min_vals_in_input_train) / (max_vals_in_input_train - min_vals_in_input_train)
input_test.head()
"""Now, let's put our training data into a **DataLoader**, which we can use to train the neural network. **DataLoaders** are great for large datasets because they make it easy to access the data in batches, make it easy to shuffle the data each epoch, and they make it easy to use a relatively small fraction of the data if we want to do a quick and dirty training for debugging our code.
To put our data training data into a **DataLoader**, we'll start by converting `input_train` into tensors with `torch.tensor()`. We'll then combine `'input_train` with `one_hot_label_train` to create a **TensorDataset**. Lastly, we'll use the **TensorDataset** to create the **DataLoader**.
**NOTE:** `torch.tensor()` will get all bent out of shape if we pass it a DataFrame directly. So, instead of passing it a DataFrame, we pass it the values by tacking `.values` on to the end of each DataFrame. We also tack on `type(torch.float32)` to make sure the numbers are saved in the correct format for the neural network to process efficiently.
"""
## Convert the DataFrame input_train into tensors
input_train_tensors = torch.tensor(input_train.values).type(torch.float32)
## now print out the first 5 rows to make sure they are what we expect.
input_train_tensors[:5]
## Convert the DataFrame input_test into tensors
input_test_tensors = torch.tensor(input_test.values).type(torch.float32)
## now print out the first 5 rows to make sure they are what we expect.
input_test_tensors[:5]
train_dataset = TensorDataset(input_train_tensors, one_hot_label_train)
train_dataloader = DataLoader(train_dataset) ## ll: add batch_size=32, shuffle=True
"""# BAM!
At long last, we have created the **DataLoaders*** that we need to train and test a neural network. Now, let's build the neural network.
----
<a id="build"></a>
# Building a neural network with multiple inputs and outputs with PyTorch and Lightning
Building a neural network with PyTorch means creating a new class. And to make it easy to train the neural network, this class will inherit from `LightningModule`.
Our new class will have the following methods:
- `__init__()` to initialize the Weights and Biases and keep track of a few other housekeeping things.
- `forward()` to make a forward pass through the neural network.
- `configure_optimizers()` to configure the optimizer. There are lots of optimizers to choose from, but in this tutorial, we'll change things up and use `Adam`.
- `training_step()` to pass the training data to `forward()`, calculate the loss and keep track of the loss values in a log file.
Also, for reference, here is a picture of the neural network we want to create:
<img src="https://github.com/StatQuest/signa/blob/main/chapter_03/images/final_nn.png?raw=1" alt="a neural network with multiple inputs and outputs" style="width: 800px;">
As we can see in the picture, our neural network has 2 inputs, one for Petal Width and one for Sepal Width, a single hidden layer with two **[ReLU](https://youtu.be/68BZ5f7P94E)** activation functions, and 3 outputs, one for each species of iris.
So, given this specification for this neural network, let's code it in a new class called `MultipleInsOuts`.
"""
class MultipleInsOuts(L.LightningModule):
    def __init__(self):
        super().__init__() ## We call the __init__() for the parent, LightningModule, so that it
                           ## can initialize itself as well.
        ## Now we the seed for the random number generorator.
        ## This ensures that when you create a model from this class, that model
        ## will start off with the exact same random numbers that I started out with when
        ## I created this demo. At least, I hope that is what happens!!! :)
        L.seed_everything(seed=42)
        ############################################################################
        ##
        ## Here is where we initialize the Weights and Biases for the neural network
        ##
        ############################################################################
        ## If you look at the drawing of the network we want to build (above),
        ## you see that we have 2 inputs that lead to 2 activation functions.
        ## We create these connections and initialize their Weights and Biases
        ## with the nn.Linear() function by setting in_features=2 and out_features=2.
        self.input_to_hidden = nn.Linear(in_features=6, out_features=2, bias=True)
        ## Next, we see that the 2 activation functions are connected to 3 outputs.
        ## We create these connections and initialize their Weights and Biases
        ## with the nn.Linear() function by setting in_features=2 and out_features=3.
        self.hidden_to_output = nn.Linear(in_features=2, out_features=2, bias=True)
        self.loss = nn.MSELoss(reduction='sum')
    def forward(self, input):
        ## First, we run the input values to the activation functions
        ## in the hidden layer
        hidden = self.input_to_hidden(input)
        ## Then we run the values through a ReLU activation function
        ## and then run those values to the output.
        output_values = self.hidden_to_output(torch.relu(hidden))
        return(output_values)
    def configure_optimizers(self):
        ## In this example, configuring the optimizer
        ## consists of passing it the weights and biases we want
        ## to optimize, which are all in self.parameters(),
        ## and setting the learning rate with lr=0.001.
        return Adam(self.parameters(), lr=0.001)
    def training_step(self, batch, batch_idx):
        ## The first thing we do is split 'batch'
        ## into the input and label values.
        inputs, labels = batch
        ## Then we run the input through the neural network
        outputs = self.forward(inputs)
        ## Then we calculate the loss.
        loss = self.loss(outputs, labels)
        ## Lastly, we could add the loss a log file
        ## so that we can graph it later. This would
        ## help us decide if we have done enough training
        ## Ideally, if we do enough training, the loss
        ## should be small and not getting any smaller.
        # self.log("loss", loss)
        return loss
model = MultipleInsOuts() # First, make model from the class
## Now print out the name and value for each named parameter
## parameter in the model. Remember parameters are variables,
## like Weights and Biases, that we can train.
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))
"""Now that we've created a class for our neural network, let's train it.
----
<a id="train"></a>
# Training our Neural Network
Training our new neural network means we create a model from the new class, `MultipleInsOuts`...
"""
model = MultipleInsOuts()
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_dataloaders=train_dataloader)
# Run the input_test_tensors through the neural network
predictions = model(input_test_tensors)
predictions[0:4,]
## Select the output with highest value...
predicted_labels = torch.argmax(predictions, dim=1) ## dim=0 applies argmax to rows, dim=1 applies argmax to columns
predicted_labels[0:4] # print out the first 4 predictions
"""In the first row index 0 had the largest value. Thus, the first prediction corresponds to **Setosa**. The second, third, and fourth rows predicted 2, which corresponds to **Virginica**.
Now, let's compare what the neural network predicted in `predicted_labels` to the known values in `label_test` and calculate the percentage of correct predictions. We do this by adding up the number of times an element in `predicted_labels` equals the corresponding element in `label_test` and dividing by the number of elements in `predicted_labels`.
"""
## Now compare predicted_labels with test_labels to calculate accuracy
## NOTE: torch.eq() computes element-wise equality between two tensors.
##       label_test, however, is just an array, so we convert it to a tensor
##       before passing it in. torch.sum() then adds up all of the "True"
##       output values to get the number of correct predictions.
##       We then divide the number of correct predictions by the number of predicted values,
##       obtained with len(predicted_labels), to get the percentage of correct predictions
label_test = label_test.iloc[:, 0]
torch.sum(torch.eq(torch.tensor(label_test), predicted_labels)) / len(predicted_labels)
"""And we see that our neural network only correctly predicts 74% of the testing data. This isn't very good. So, will training our model for more epochs improve the model's predictions?
One way to answer that question is to just train for longer and see what happens.
The good news is that because we're using **Lightning**, we can pick up where we left off training without starting over from scratch. This is because training with **Lightning** creates _checkpoint_ files that keep track of the Weights and Biases as they change. As a result, all we have to do to pick up where we left off is tell the `Trainer` where the checkpoint files are. This is awesome and will save us a lot of time since we don't have to retrain the first **10** epochs. So, let's add an additional **90** epochs to the training.
To add additional epochs to the training, we first identify where the checkpoint file is with the following command.
"""
path_to_checkpoint = trainer.checkpoint_callback.best_model_path ## By default, "best" = "most recent"
## First, create a new Lightning Trainer
trainer = L.Trainer(max_epochs=100) # Before, max_epochs=10, so, by setting it to 100, we're adding 90 more.
## Then call trainer.fit() using the path to the most recent checkpoint files
## so that we can pick up where we left off.
trainer.fit(model, train_dataloaders=train_dataloader, ckpt_path=path_to_checkpoint)
# Run the input_test_tensors through the neural network
predictions = model(input_test_tensors)
## Select the output with highest value...
predicted_labels = torch.argmax(predictions, dim=1) ## dim=0 applies softmax to rows, dim=1 applies softmax to columns
## Now compare predicted_labels with test_labels to calculate accuracy
## NOTE: torch.eq() computes element-wise equality between two tensors.
##       label_test, however, is just an array, so we convert it to a tensor
##       before passing it in. torch.sum() then adds up all of the "True"
##       output values to get the number of correct predictions.
##       We then divide the number of correct predictions by the number of predicted values,
##       obtained with len(predicted_labels), to get the percentage of correct predictions
torch.sum(torch.eq(torch.tensor(label_test), predicted_labels)) / len(predicted_labels)
"""After 100 training epochs, we correctly classified 92% of the testing data. This means adding more training was helpful!
# Double BAM!!
----
<a id="predict"></a>
# Make a Prediction with New Data
Now that our model is trained, we can use it to make predictions from new data. This is done by passing the model a tensor with normalized petal and sepal widths wrapped up in a tensor.
For example, if the raw petal and sepal width measurements were 0.2 and 3.0, we would first normalize them using the maximum and minimum values we calculated with the training data.
"""
normalized_values = ([0.15374502201420875,0.4499111294211043,0.5495266686456031,7.391934074578481,4.737483207307689,0] - min_vals_in_input_train) / (max_vals_in_input_train - min_vals_in_input_train)
normalized_values
torch.argmax(model(torch.tensor(normalized_values).type(torch.float32)))
"""And first output has the largest value, meaning that the neural network predicts that the measurements come from **Setosa**.
# TRIPLE BAM!!!
"""
model.eval()
with torch.no_grad():
    logits = model(input_test_tensors)                 # (N, 2)
    probs = F.softmax(logits, dim=1)                   # (N, 2)
    y_score = probs[:, 1].cpu().numpy()                # 正类概率 (N,)

fpr, tpr, _ = roc_curve(label_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Binary, positive class=1)")
plt.legend(loc="lower right")
plt.savefig("./roc.png", dpi=200)
