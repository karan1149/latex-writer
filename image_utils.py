import os 
import numpy as np 
from PIL import Image
import random

### TENSORPORT SUPPORT

from tensorport import get_data_path

images_location = get_data_path(local_root=".", dataset_name="karan1149/math-data2", local_repo="extracted_images", path="")

###

random.seed(42)

possible_classes = [name for name in os.listdir(images_location) if os.path.isdir(os.path.join(images_location, name))]

desired_symbols = set(["-", "+", "x", "0", '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '='])
NUM_CLASSES = len(desired_symbols)

possible_classes_totals = []
for possible_class in possible_classes:
	names = [name for name in os.listdir(os.path.join(images_location, possible_class)) if not name.startswith(".")]
	possible_classes_totals.append((possible_class, len(names)))

possible_classes_totals = sorted(possible_classes_totals, key=lambda x: x[0])
possible_classes_totals = [total for total in possible_classes_totals if total[0] in desired_symbols]

print(possible_classes_totals, len(possible_classes_totals), sum(possible_class[1] for possible_class in possible_classes_totals))
assert(len(possible_classes_totals) == len(desired_symbols))

def format_np_array(images):
	images = (np.array(images).astype(np.float32) - 127.5) / 127.5
	images = images.reshape((-1, 28, 28))
	images = np.expand_dims(images, axis=1)
	return images

def get_math_dataset():
	max_images = 2909
	labeled_images = []
	classes_counts = []
	for i, symbol_pair in enumerate(possible_classes_totals):
		filenames = [os.path.join(images_location, symbol_pair[0], name) for name in os.listdir(os.path.join(images_location, symbol_pair[0])) if not name.startswith(".")]
		random.shuffle(filenames)
		filenames = filenames[:2909]
		images = [np.array(Image.open(filename).convert("L").getdata()) for filename in filenames]
		curr_labeled_images = zip(images, [i] * len(images))
		classes_counts.append(len(images))
		labeled_images.extend(curr_labeled_images)
	print(classes_counts)

	random.shuffle(labeled_images)

	print(np.array(labeled_images).shape)
	print(labeled_images[0][0].shape)

	x, y = zip(*labeled_images)
	x = format_np_array(x)
	y = np.array(y)
	print(x.shape, y.shape)

	test_x = x[:int(x.shape[0] / 10)]
	train_x = x[int(x.shape[0] / 10):]

	test_y = y[:int(y.shape[0] / 10)]
	train_y = y[int(y.shape[0] / 10):]

	classes_min = min(classes_counts)
	classes_distribution = np.array(classes_counts) / classes_min

	print("distribution:", classes_distribution)

	return train_x, train_y, test_x, test_y, classes_distribution

get_math_dataset()

