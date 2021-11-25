# TODO Step 2: Add target and patience arguments to the argument parser
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')

# TODO Step 2: Define the StopAtAccuracy callback
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target=0.85, patience=2):
        self.target = target
        self.patience = patience
        self.stopped_epoch = 0
        self.met_target = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.target:
            self.met_target += 1
        else:
            self.met_target = 0
            
        if self.met_target >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Early stopping after epoch {}'.format(self.stopped_epoch + 1))

# TODO Step 2: Add the StopAtAccuracy callback to the callbacks list
StopAtAccuracy(target=args.target_accuracy, patience=args.patience)
