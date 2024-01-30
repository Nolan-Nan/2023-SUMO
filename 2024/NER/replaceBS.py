import spacy
from spacy.training.example import Example
from sklearn.model_selection import train_test_split
import json

# Assuming your JSON file is named 'your_file.json'
file_path = 'NER_TRAIN/NER_TRAIN_JUDGEMENT.json'

# Read JSON data from the file
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Add entity labels to the NER component
entity_labels = ["COURT", "PETITIONER", "RESPONDENT", "JUDGE", "LAWYER", "DATE", "ORG", "GPE", "STATUTE", "PROVISION", "PRECEDENT", "CASE_NUMBER", "WITNESS", "OTHER_PERSON"]

for label in entity_labels:
    ner = nlp.get_pipe("ner")
    ner.add_label(label)

# Convert JSON data to spaCy training format
train_data = []
for item in json_data:
    text = item.get("data", {}).get("text", "")
    entities = []

    for annotation in item["annotations"][0]["result"]:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        label = annotation["value"]["labels"][0]
        entities.append((start, end, label))

    train_data.append((text, {"entities": entities}))

# Split the data into train and test sets
train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=42)

# Train the NER model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    for epoch in range(10):  # Adjust the number of epochs
        print(epoch)
        for example in train_set:
            doc = nlp.make_doc(example[0])
            gold_dict = Example.from_dict(doc, example[1])
            nlp.update([gold_dict], drop=0.5)  # Adjust the dropout value

# Evaluate the model on the test set
docs = []
gold_dicts = []

for example in test_set:
    text, annotations = example
    doc = nlp(text)
    print("Text:", example[0])
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print("-----")
    gold_dict = Example.from_dict(doc, annotations).to_dict(annot=True)

    docs.append(doc)
    gold_dicts.append(gold_dict)

# Compute precision, recall, and F1 score using spaCy's scorer
scorer = nlp.evaluate(docs, gold_dicts)
print("Precision:", scorer.precision)
print("Recall:", scorer.recall)
print("F1 score:", scorer.fscore)
nlp.to_disk("ner_model")
