
import os

# get all the classes in the dataset
root_dirpath = 'C:\\Users\\Dell\\Downloads\\Dataset'
annotations_dirpath = os.path.join(root_dirpath, 'annotations/Annotation')
images_dirpath = os.path.join(root_dirpath, 'images/Images')

ID2LABEL = {}
LABEL2ID = {}
for idx, image_filename in enumerate(os.listdir(images_dirpath)):
    label = image_filename.split('-')[1].lower()
    ID2LABEL[idx] = label
    LABEL2ID[label] = idx

NUM_LABELS = len(ID2LABEL)
print(f"NUM_LABELS: {NUM_LABELS}\n")
import PIL
from tqdm.notebook import tqdm

# let's import the dataset
labels = []
images = []
for image_dirpath in tqdm(os.listdir(annotations_dirpath)):
    for image_filepath in os.listdir(os.path.join(annotations_dirpath, image_dirpath)):
        annotation_filepath = os.path.join(annotations_dirpath, image_dirpath, image_filepath)
        image_filepath = os.path.join(images_dirpath, image_dirpath, image_filepath + '.jpg')

        # getting label from directory name instead of xml for speed
        label = image_dirpath.split('-')[1].lower()
        # use lazy loading object to reduce ram usage
        with PIL.Image.open(image_filepath) as image:
            # only get images that are jpegs
            if isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
                labels.append(label)
                images.append(image_filepath)

print(f"NUMBER OF LABELS: {len(labels)}")
print(f"NUMBER OF IMAGES: {len(images)}")
import datasets

# let's create the raw datasets of dog breeds
raw_datasets = datasets.Dataset.from_dict(
    mapping = {'image': images, 'labels': labels,},
    features = datasets.Features({
        'image': datasets.Image(),
        'labels': datasets.features.ClassLabel(names=list(LABEL2ID.keys())),
    })
)

raw_datasets
import random
import torch
import numpy as np

# we set the seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
import matplotlib.pyplot as plt


# get random dog image indexes for plotting
random_idxs = random.sample(range(len(labels)), 9)

# plot random dog image in 3 by 3 grid
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
fig.suptitle('Dog Breeds Image Examples')

for idx,ax in zip(random_idxs, axs.flat):
    image = raw_datasets[idx]['image']
    label_id = raw_datasets[idx]['labels']
    label = ID2LABEL[label_id]
    ax.axis('off')
    ax.set_title(f"{idx}:{label}")
    ax.imshow(image)
    images[0]

plt.show()
# get label counts sorted by counts in descending order
label_counts = {}
for label in labels:
    label_counts[label] = label_counts.get(label,0)+1
    # sort by count
label_counts = dict(sorted(label_counts.items(), key=lambda kv: kv[1]))

# plot counts by dog breed
# there's around 150-250 images of each dog breed
fig, ax = plt.subplots(figsize=(10,20))
fig.suptitle('Count by Dog Breed')
ax.barh(list(label_counts.keys()), list(label_counts.values()))
plt.show()
from transformers import AutoImageProcessor

# clone image processor from huggingface hub
CHECKPOINT = 'google/vit-base-patch16-224-in21k'
image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
from datasets import DatasetDict

# split dataset into train-validation-test split (80%-10%-10%)
train_test_datasets = raw_datasets.train_test_split(test_size=0.2, seed=SEED, shuffle=True)
validation_test_datasets = train_test_datasets['test'].train_test_split(test_size=0.5, seed=SEED, shuffle=True)

# create clean dataset with splits
datasets = DatasetDict({
    'train': train_test_datasets['train'],
    'validation': validation_test_datasets['train'],
    'test': validation_test_datasets['test'],
})

datasets
# we use set_transform to do on-the-fly processing of images to reduce ram usage
def train_transforms(example):
    example.update(image_processor(example['image'], return_tensors='pt'))
    return example

def eval_transforms(example):
    example.update(image_processor(example['image'], return_tensors='pt'))
    return example

datasets['train'].set_transform(train_transforms)
datasets['validation'].set_transform(eval_transforms)
datasets['test'].set_transform(eval_transforms)
from transformers import AutoModelForImageClassification

# let's clone model from huggingface hub
model = AutoModelForImageClassification.from_pretrained(CHECKPOINT, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
import numpy as np
import evaluate
from sklearn.metrics import top_k_accuracy_score

# let's setup the training metrics with topk-accuracies and macro f1
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    top3_accuracy = top_k_accuracy_score(labels, logits, k=3)
    top5_accuracy = top_k_accuracy_score(labels, logits, k=5)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')
    return {
        **accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        **f1,
    }
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
from transformers import TrainingArguments, Trainer

# we setup the training configurations
#                        seed: we use this for reproducibility
#                  output_dir: we output the model checkpoints to this directory
#                       optim: we use the updated adamw optimizer
#            num_train_epochs: we train for 3 epochs
# per_device_train_batch_size: increase batch size to speed up training
#  per_device_eval_batch_size: increase batch size to speed up evaluation
#               save_strategy: we save the model on each epoch instead of every 500 steps
#         evaluation_strategy: we evaluate the model on each epoch instead of every 500 steps
#      load_best_model_at_end: load the best model with lowest validation loss
#                   report_to: suppress default reporting to third-party loggers
#       remove_unused_columns: this is set False to allow on-the-fly preprocessing
training_args = TrainingArguments(
    seed=SEED,
    output_dir='./results',
    optim='adamw_torch',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    report_to='none',
    remove_unused_columns=False,
)

# we setup the trainer with our splitted datasets
# with custom metrics and collate function, and on-the-fly preproccessing
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    data_collator=collate_fn,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)
# let's get the unfinetuned performance of the model on the test set as a baseline
trainer.evaluate(datasets['test'])
trainer.train()
trainer.evaluate(datasets['test'])
import requests
from transformers import pipeline

# let's create a pipeline to classifiy dog images into their respective dog breeds
dog_breeds_multiclass_image_classifier = pipeline("image-classification", model=model.to(torch.device('cpu')), image_processor=image_processor)