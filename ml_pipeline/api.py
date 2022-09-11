import pandas as pd
from fastapi import FastAPI
from metaflow import Flow
from pydantic import BaseModel
from .preprocess import add_features, vectorize, EXTRA_FEATURES

# Load the model on startup from Metaflow
# Could also get it from MLflow and/or build a docker image
run = Flow("TrainFlow").latest_successful_run
print("Using run: %s" % str(run))
best_clf = run.data.best_clf
vectorizer = run.data.vectorizer
names = vectorizer.get_feature_names_out()
vocab_size = len(names)

app = FastAPI()

class PredictInput(BaseModel):
  text: str
  keyword: str = ""

@app.post("/predict")
def read_item(input: PredictInput):
  print(input)
  df = pd.DataFrame({"text": [input.text], "keyword": [input.keyword]})
  add_features(df)
  vector = vectorize(df, vectorizer)
  print([(idx, names[idx]) for idx in vector.nonzero()[1] if idx < vocab_size])
  # not using the word count/length in the model, just inspecting
  print(df[EXTRA_FEATURES + ["count.words", "average.word.length"]])
  pred = int(best_clf.predict(vector)[0])
  return {"predict": pred}
