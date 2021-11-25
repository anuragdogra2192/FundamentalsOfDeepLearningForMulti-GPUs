# TODO Step 1: Define the PrintThroughput callback
class PrintThroughput(tf.keras.callbacks.Callback):
    def __init__(self, total_images=0):
        self.total_images = total_images
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time()
    
    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.epoch_start_time
        images_per_sec = round(self.total_images / epoch_time, 2)
        print('Images/sec: {}'.format(images_per_sec))

# TODO Step 1: Add the PrintThrought callback to the callbacks list
PrintThroughput(total_images=len(y_train))
