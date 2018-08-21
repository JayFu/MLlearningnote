import math

import tensorflow as tf
from tensorflow.python.data import Dataset
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_houhlsing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe
california_housing_dataframe.describe()

# 特征输入值是总房间数
my_feature = california_housing_dataframe[["total_rooms"]]
# 特征列
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# 定义标签
targets = california_housing_dataframe["median_house_value"]

# 梯度下降法来优化模型
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001) #学习效率
# 把梯度裁剪应用到这个优化器中
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# 建立一个线性回归模型应用到特征列和优化器中
# 设定学习效率用于监督学习
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

# def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
#     """Trains a linear regression model of one feature.
  
#     Args:
#       features: pandas DataFrame of features
#       targets: pandas DataFrame of targets
#       batch_size: Size of batches to be passed to the model
#       shuffle: True or False. Whether to shuffle the data.
#       num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
#     Returns:
#       Tuple of (features, labels) for next data batch
#     """
  
#     # 把一个pd数据转化成np数组字典
#     features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
#     # 构建数据集，构建单批次和迭代
#     ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
#     ds = ds.batch(batch_size).repeat(num_epochs)
    
#     # 对数据进行洗牌操作
#     if shuffle:
#       ds = ds.shuffle(buffer_size=10000)
    
#     # 进入下一批次数据
#     features, labels = ds.make_one_shot_iterator().get_next()
#     return features, labels

# _ = linear_regressor.train(  
#     input_fn = lambda:my_input_fn(my_feature, targets),  
#     steps=100#训练100步
# )

# # 创建一个输入函数用于预测 
# prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)  
  
# # 在线性回归模型上调用预测函数进行预测
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)  
  
# # 将预测结果格式化为一个np数组，这样可以看见错误  
# predictions = np.array([item['predictions'][0] for item in predictions])  
  
# # 均方误差和均方根误差，用于评估模型
# mean_squared_error = metrics.mean_squared_error(predictions, targets)  
# root_mean_squared_error = math.sqrt(mean_squared_error)  
# print("Mean Squared Error (on training data): %0.3f" % mean_squared_error) 
# print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

# # 对比预测与目标
# calibration_data = pd.DataFrame()  
# calibration_data["predictions"] = pd.Series(predictions)  
# calibration_data["targets"] = pd.Series(targets)  
# calibration_data.describe()  

def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
      """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns
    feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
    # Create input functions
    training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
     # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

  # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range (0, periods):
    # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
    
        # Compute loss.
        root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])
    
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents, sample[my_feature].max()), sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1
)