import tensorflow as tf
tf.reset_default_graph()
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import itertools
#from subprocess import check_output
#print(check_output(["ls", "."]).decode("utf8"))


# Description of the Data fields
# id - a unique identifier for each trip
# vendor_id - a code indicating the provider associated with the trip record
# pickup_datetime - date and time when the meter was engaged
# dropoff_datetime - date and time when the meter was disengaged
# passenger_count - the number of passengers in the vehicle (driver entered value)
# pickup_longitude - the longitude where the meter was engaged
# pickup_latitude - the latitude where the meter was engaged
# dropoff_longitude - the longitude where the meter was disengaged
# dropoff_latitude - the latitude where the meter was disengaged
# store_and_fwd_flag - This flag indicates whether the trip record was held in 
#                      vehicle memory before sending to the vendor because the 
#                      vehicle did not have a connection to the server - 
#                       Y=store and forward; N=not a store and forward trip
# trip_duration - duration of the trip in seconds

tf.logging.set_verbosity(tf.logging.INFO)

# The Columns & label we would be using. 
COLUMNS = ["vendor_id","pickup_datetime","passenger_count","pickup_longitude",\
"pickup_latitude","dropoff_longitude",\
"dropoff_latitude","store_and_fwd_flag"]

categorical_features = ["store_and_fwd_flag", "pickup_datetime"]

continuous_features = ["vendor_id", "passenger_count", "pickup_longitude",
           "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]

LABEL_COLUMN = "trip_duration"

train_df = []
test_df =[] 
predict_df =[]

def input_fn(df, training = True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_features}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in categorical_features}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label
    
    # Returns the feature columns    
    return feature_cols

def train_input_fn():
    return input_fn(train_df)

def eval_input_fn():
    return input_fn(test_df)

def pred_input_fn():
    return input_fn(predict_df, False)

def main():
	global train_df
	global test_df
	global predict_df

	# Load datasets
	df = pd.read_csv("train.csv")

	# Split data to training and testing set.
	train_df, test_df = train_test_split(df, test_size = 0.2)

	#load the prediction set
	predict_df = pd.read_csv("test.csv")
	
	# print (predict_df.head(20))
	#predict_df['pickup_datetime'] = predict_df['pickup_datetime'].apply(str)
	#predict_df['vendor_id'] = pd.to_numeric(predict_df['vendor_id'])
	
	# print ('train_set shape {}'.format(train_df.shape))
	# print ('test_set shape {}'.format(test_df.shape))
	# print ('prediction_set shape {}'.format(predict_df.shape))
		
	# print("Training types:")
	# print (train_df.dtypes)
	# print("Predict types:")
	# print (predict_df.dtypes)

  	engineered_features = []

  	for continuous_feature in continuous_features:
  		engineered_features.append(
  			tf.contrib.layers.real_valued_column(continuous_feature))

  	for categorical_feature in categorical_features:
  		sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
  			categorical_feature, hash_bucket_size=1000)
  		engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=9,
                                                                  combiner="sum"))
  	#Build 2 layer fully connected DNN with 10, 10 units respectively.
  	regressor = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
  		hidden_units=[10, 10],
  		model_dir="./nyc_model")

  	# Train
  	regressor.fit(input_fn=train_input_fn, max_steps=1000)

  	# Evaluate our Model
  	print('Evaluating the model...')
  	results = regressor.evaluate(input_fn=eval_input_fn, steps=1)
  	for key in sorted(results):
  		print("%s: %s" % (key, results[key]))

  	# Predict.
  	y = regressor.predict(input_fn=pred_input_fn, outputs=None)
   
   	#.predict() returns an iterator of dicts; convert to a list and print predictions
   	submission = ['id,trip_duration']
   	ids = predict_df["id"].values
	for prediction, id in zip(y, ids):
			submission.append('{0},{1}'.format(id, int(prediction)))
		
 	submission = '\n'.join(submission)
 	with open('submission-dovik.csv', 'w') as outfile:
 		outfile.write(submission)


if __name__ == "__main__":
	main()
