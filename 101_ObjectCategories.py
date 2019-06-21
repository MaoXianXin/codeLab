# USAGE
# python train_auto_keras.py

# import the necessary packages
from sklearn.metrics import classification_report
# from tensorflow.keras.datasets import cifar100
import autokeras as ak
from autokeras.image.image_supervised import load_image_dataset
import os

# def main():
# initialize the output directory
OUTPUT_PATH = "output"

# initialize the list of trianing times that we'll allow
# Auto-Keras to train for
TRAINING_TIMES = [
	# 60 * 60,		# 1 hour
	# 60 * 60 * 2,	# 2 hours
	# 60 * 60 * 4,	# 4 hours
	# 60 * 60 * 8,	# 8 hours
	60 * 60 * 12,	# 12 hours
	# 60 * 60 * 24,	# 24 hours
]

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading 101_ObjectCategories data...")
trainX, trainY = load_image_dataset(csv_file_path='./101_ObjectCategories/train/label.csv',
									  images_path='./101_ObjectCategories/train')
# testX, testY = load_image_dataset(csv_file_path='./101_ObjectCategories/test/label.csv',
# 									images_path='./101_ObjectCategories/test')
# trainX = trainX.astype("float") / 255.0
# testX = testX.astype("float") / 255.0

# initialize the label names for the CIFAR-10 dataset
# labelNames = ['scorpion', 'hedgehog', 'anchor', 'joshua_tree', 'lobster', 'pyramid', 'ant',
# 			  'dollar_bill', 'gerenuk', 'bass', 'trilobite', 'chandelier', 'dolphin', 'panda',
# 			  'inline_skate', 'gramophone', 'camera', 'pizza', 'kangaroo', 'crocodile_head',
# 			  'flamingo', 'hawksbill', 'ceiling_fan', 'lamp', 'lotus', 'ketch', 'cannon', 'stapler',
# 			  'crocodile', 'wrench', 'scissors', 'umbrella', 'BACKGROUND_Google', 'minaret', 'crayfish',
# 			  'ewer', 'cougar_body', 'okapi', 'Motorbikes', 'accordion', 'sunflower', 'headphone', 'bonsai',
# 			  'mayfly', 'elephant', 'airplanes', 'cup', 'garfield', 'platypus', 'brontosaurus', 'butterfly',
# 			  'euphonium', 'grand_piano', 'stegosaurus', 'nautilus', 'helicopter', 'rhino', 'wheelchair',
# 			  'pagoda', 'laptop', 'tick', 'schooner', 'yin_yang', 'beaver', 'snoopy', 'watch', 'emu', 'dragonfly',
# 			  'dalmatian', 'revolver', 'ibis', 'starfish', 'binocular', 'stop_sign', 'mandolin', 'octopus', 'pigeon',
# 			  'sea_horse', 'menorah', 'wild_cat', 'ferry', 'flamingo_head', 'saxophone', 'metronome', 'brain',
# 			  'car_side', 'crab', 'Leopards', 'cougar_face', 'electric_guitar', 'cellphone', 'Faces_easy',
# 			  'windsor_chair', 'Faces', 'water_lilly', 'rooster', 'llama', 'chair', 'strawberry', 'barrel',
# 			  'buddha', 'soccer_ball'
# 	]

# loop over the number of seconds to allow the current Auto-Keras
# model to train for
for seconds in TRAINING_TIMES:
	# train our Auto-Keras model
	print("[INFO] training model for {} seconds max...".format(
		seconds))
	model = ak.ImageClassifier(verbose=True)
	model.fit(trainX, trainY, time_limit=seconds)
	model.final_fit(trainX, trainY, trainX, trainY, retrain=True)

	# evaluate the Auto-Keras model
	score = model.evaluate(trainX, trainY)
	print(score)
	model.export_autokeras_model('101_Objects.h5')
	# predictions = model.predict(trainX)
	# report = classification_report(trainY, predictions,
	# 	target_names=labelNames)

	# write the report to disk
	# if not os.path.exists(OUTPUT_PATH):
	# 	os.mkdir(OUTPUT_PATH)
	# p = os.path.join(OUTPUT_PATH, "{}.txt".format(seconds))
	# f = open(p, "w")
	# f.write(report)
	# f.write("\nscore: {}".format(score))
	# f.close()

# if this is the main thread of execution then start the process (our
# code must be wrapped like this to avoid threading issues with
# TensorFlow)
# if __name__ == "__main__":
# 	main()