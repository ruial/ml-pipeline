from fastapi import FastAPI
from metaflow import Flow
from pydantic import BaseModel

import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode("utf-8")

# Load the model on startup from Metaflow
# Could also get it from MLflow and/or build a docker image
run = Flow("TrainFlow").latest_successful_run
print("Using run: %s" % str(run))
best_clf = run.data.best_clf
vectorizer = run.data.vectorizer
names = vectorizer.get_feature_names_out()

app = FastAPI()

class PredictInput(BaseModel):
    text: str

@app.post("/predict")
def read_item(input: PredictInput):
  text = remove_accents(input.text)
  vector = vectorizer.transform([text])
  print(text)
  print([(idx, names[idx]) for idx in vector.nonzero()[1]])
  pred = int(best_clf.predict(vector)[0])
  return {"predict": pred}
