# Setup plotting
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Read in data set from CSV
red_wine = pd.read_csv('../data/red-wine.csv')
print(red_wine.head())

# View the number of rows and columns of a dataframe.
print(red_wine.shape) # (rows = data points, columns = row attributes)

# The target attribute is 'quality', and the remaining columns are the features.
input_shape = [11] # 11 is the input shape because there are 11 features.


# Now define your model using keras
model = keras.Sequential([
    layers.Dense(units=512, activation="relu" input_shape=[11]),
    layers.Dense(units=512, activation="relu"),
    layers.Dense(units=512, activation="relu"),
    # the linear output layer
    layers.Dense(units=1)
])

# Determine the models SGD optimizer for the learning rate and the loss function.
model.compile(
    optimizer="adam",
    loss="mae",
)

# A model's weights are kept in its weights attribute as a list of tensors.
w, b = model.weights

print("Weights\n{}\n\nBias\n{}".format(w,b))

# Training your model

history = model.fit(
    X_train, y_train, # training data
    validation_data=(X_valid, y_valid), # validation data
    batch_size=256, # process 256 training points and then use SGD to alter weights and biases
    epoch=10, # run through the dataset 10 times
)

# Can use a pandas Dataframe to show the loss progression
history_df = pd.DataFrame(history.history)
history_df["loss"].pliot()
