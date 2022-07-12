from metaflow import FlowSpec, step, card, Parameter, current, S3
from metaflow.cards import Image
import mlflow

class TrainFlow(FlowSpec):

  train_file = Parameter(
    "train_file", help="Train dataset", default="datasets/twitter-train.csv"
  )

  split_size = Parameter(
    "split_size", help="Split size", default=0.2
  )

  nb_alpha = Parameter(
    "nb_alpha", help="Naive Bayes alpha hyperparameters", default="0.1, 0.2, 0.5, 0.8, 0.9, 1"
  )

  mlflow_experiment = Parameter(
    "mlflow_experiment", help="MLflow experiment name", default="twitter-experiment"
  )


  @step
  def start(self):
    import pandas as pd
    from io import StringIO
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    with S3(s3root="s3://ml-bucket") as s3:
      s3obj = s3.get(self.train_file)
      print("Object found at", s3obj.url)
      self.train_df = pd.read_csv(StringIO(s3obj.text))

    self.train_df["text"] = self.train_df["text"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    self.train_df = self.train_df.drop_duplicates(subset=['text'])
    print("Split size: %f" % self.split_size)
    X_train, X_test, y_train, y_test = train_test_split(self.train_df, self.train_df['target'],
      test_size=self.split_size, random_state=1, stratify=self.train_df['target'])
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    stop_words = ENGLISH_STOP_WORDS.union(["amp", "gt", "rt", "like", "just", "http", "https"])
    self.vectorizer = CountVectorizer(min_df=2, stop_words=stop_words)
    self.train_vectors = self.vectorizer.fit_transform(X_train["text"])
    self.test_vectors = self.vectorizer.transform(X_test["text"])

    counts = pd.DataFrame(self.train_vectors.toarray(), columns=self.vectorizer.get_feature_names_out())
    print(counts.sum().sort_values(ascending=False).head(10))
  
    print("Tracking uri: ", mlflow.get_tracking_uri())
    experiment = mlflow.set_experiment(self.mlflow_experiment)
    self.experiment_id = experiment.experiment_id

    self.nb_alpha_params = self.nb_alpha.split(",")
    self.next(self.train, foreach="nb_alpha_params")
  

  @step
  def train(self):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import f1_score
    from mlflow.models.signature import infer_signature
    import pickle
    import os

    self.alpha = float(self.input)
    print("Training NB with alpha %f" % self.alpha)
    self.clf = MultinomialNB(alpha=self.alpha)
    self.clf.fit(self.train_vectors, self.y_train)
    self.score = f1_score(self.y_test, self.clf.predict(self.test_vectors))
    print(f"Alpha: {self.alpha}, F1 score: {self.score}")

    with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{current.flow_name}_{current.run_id}_{current.task_id}") as run:
      mlflow.log_param("alpha", self.alpha)
      mlflow.log_metric("F1", self.score)
      signature = infer_signature(self.train_vectors, self.clf.predict(self.train_vectors))
      # an sklearn pipeline could probably be logged instead of having a separate vectorizer
      mlflow.sklearn.log_model(self.clf, "nb_classifier", signature=signature)
      vectorizer_path = f"vectorizer-{run.info.run_id}.pickle"
      with open(vectorizer_path, "wb") as handle:
        pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(vectorizer_path)
      os.remove(vectorizer_path)

    self.next(self.join)

  @step
  def join(self, inputs):
    scores = [input.score for input in inputs]
    max_score = max(scores)
    max_index = scores.index(max_score)
    self.best_clf = inputs[max_index].clf
    # since accessing the vectorizer in the join because of the mlflow logging, can't include it in merge_artifacts
    self.vectorizer = inputs[max_index].vectorizer
    print("Max F1:", max_score, "Alpha:", self.best_clf.alpha)
    self.merge_artifacts(inputs, include=['test_vectors', 'y_test'])
    self.next(self.end)

  @card(type='blank')
  @step
  def end(self):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(self.best_clf.predict(self.test_vectors), self.y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.best_clf.classes_)
    disp.plot()
    current.card.append(Image.from_matplotlib(disp.figure_))
    print("TrainFlow is done.")


if __name__ == "__main__":
  TrainFlow()
