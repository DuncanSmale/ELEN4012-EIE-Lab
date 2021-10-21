import visualkeras
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model("models/TESTLLRVoteRange.h5")
visualkeras.layered_view(
    model, legend=True, to_file='output.png', scale_xy=2, scale_z=1, max_z=10).show()  # write and show
model2 = tf.keras.models.load_model("models/TESTLLRVoteRangeNEW.h5")
visualkeras.layered_view(
    model, legend=True, to_file='output2.png', scale_xy=2, scale_z=1, max_z=10).show()  # write and show
