import os 

def get_math_dataset():
	images_location = "extracted_images"
	possible_classes = [name for name in os.listdir(images_location) if os.path.isdir(os.path.join(images_location, name))]
	possible_classes_totals = []
	for possible_class in possible_classes:
		names = [name for name in os.listdir(os.path.join(images_location, possible_class)) if not name.startswith(".")]
		possible_classes_totals.append((possible_class, len(names)))
	possible_classes_totals = sorted(possible_classes_totals, key=lambda x: x[1], reverse=True)
	print(possible_classes_totals, len(possible_classes_totals), sum(possible_class[1] for possible_class in possible_classes_totals))
	possible_classes_totals = [total for total in possible_classes_totals if total[1] > 2000]
	print()
	print(possible_classes_totals, len(possible_classes_totals), sum(possible_class[1] for possible_class in possible_classes_totals))

get_math_dataset()