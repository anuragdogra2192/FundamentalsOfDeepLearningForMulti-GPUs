# STEP 1: Import the csv library
import csv

# STEP 2: Read through, and then copy over the custom callback
class SaveTrainingData(tf.keras.callbacks.Callback):
    def __init__(self, data_filepath=''):
        self.data_filepath = data_filepath

    def on_train_begin(self, logs=None):       
        file = open(self.data_filepath, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['time', 'val_accuracy'])
        writer.writerow([0.0, 0.0])
        file.close()  

        self.train_start_time = time()

    def on_epoch_end(self, epoch, logs={}):
        total_time = time() - self.train_start_time
        file = open(self.data_filepath, 'a')
        writer = csv.writer(file)
        writer.writerow([round(total_time,1), round(logs['val_accuracy'], 4)])
        file.close()

# STEP 3: Create the CSV filename from the hyperparameters and add the callback
data_filepath = "training_data/{}ranks-{}bs-{}lr.csv".format(hvd.size(), args.batch_size, args.base_lr)
callbacks.append(SaveTrainingData(data_filepath=data_filepath))
