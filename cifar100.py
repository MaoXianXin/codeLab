# USAGE
# python train_auto_keras.py

# import the necessary packages
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import cifar100
import autokeras as ak
import os

def main():
	# initialize the output directory
	OUTPUT_PATH = "output"

	# initialize the list of trianing times that we'll allow
	# Auto-Keras to train for
	TRAINING_TIMES = [
		60 * 60,		# 1 hour
		60 * 60 * 2,	# 2 hours
		60 * 60 * 4,	# 4 hours
		60 * 60 * 8,	# 8 hours
		60 * 60 * 12,	# 12 hours
		60 * 60 * 24,	# 24 hours
	]

	# load the training and testing data, then scale it into the
	# range [0, 1]
	print("[INFO] loading CIFAR-100 data...")
	((trainX, trainY), (testX, testY)) = cifar100.load_data()
	trainX = trainX.astype("float") / 255.0
	testX = testX.astype("float") / 255.0

	# initialize the label names for the CIFAR-10 dataset
	labelNames = [
		'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
		'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
		'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
		'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
		'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
		'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
		'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
		'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
		'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
		'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
		'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
		'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
		'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
		'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
		'worm'
	]

	# loop over the number of seconds to allow the current Auto-Keras
	# model to train for
	for seconds in TRAINING_TIMES:
		# train our Auto-Keras model
		print("[INFO] training model for {} seconds max...".format(
			seconds))
		model = ak.ImageClassifier(verbose=True)
		model.fit(trainX, trainY, time_limit=seconds)
		model.final_fit(trainX, trainY, testX, testY, retrain=True)

		# evaluate the Auto-Keras model
		score = model.evaluate(testX, testY)
		predictions = model.predict(testX)
		report = classification_report(testY, predictions,
			target_names=labelNames)

		# write the report to disk
		if not os.path.exists(OUTPUT_PATH):
			os.mkdir(OUTPUT_PATH)
		p = os.path.join(OUTPUT_PATH, "{}.txt".format(seconds))
		f = open(p, "w")
		f.write(report)
		f.write("\nscore: {}".format(score))
		f.close()

# if this is the main thread of execution then start the process (our
# code must be wrapped like this to avoid threading issues with
# TensorFlow)
if __name__ == "__main__":
	main()