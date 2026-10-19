"""Shared constants for labels and ImageNet preprocessing."""

CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
