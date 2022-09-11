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
    "nb_alpha", help="Naive Bayes alpha hyperparameters", default="0.3, 0.6, 0.9"
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
    from preprocess import add_features, vectorize

    with S3(s3root="s3://ml-bucket") as s3:
      s3obj = s3.get(self.train_file)
      print("Object found at", s3obj.url)
      self.train_df = pd.read_csv(StringIO(s3obj.text))

    add_features(self.train_df)
    self.train_df = self.train_df.drop_duplicates(subset=["text"])
    print("Split size: %f" % self.split_size)
    X_train, X_test, y_train, y_test = train_test_split(self.train_df, self.train_df["target"],
      test_size=self.split_size, random_state=1, stratify=self.train_df["target"])
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test

    stop_words = ENGLISH_STOP_WORDS.union(["http", "https"])
    self.vectorizer = CountVectorizer(min_df=3, stop_words=stop_words)

    count_vectors = self.vectorizer.fit_transform(X_train["text"])
    counts = pd.DataFrame(count_vectors.toarray(), columns=self.vectorizer.get_feature_names_out())
    print(counts.sum().sort_values(ascending=False).head(10))

    self.train_vectors = vectorize(X_train, self.vectorizer)
    self.test_vectors = vectorize(X_test, self.vectorizer)
  
    print("Tracking uri: ", mlflow.get_tracking_uri())
    experiment = mlflow.set_experiment(self.mlflow_experiment)
    self.experiment_id = experiment.experiment_id

    # this algorithm is fast to train, but parallelizing hyperparameter tuning can be helpful in other models
    # also check https://github.com/optuna/optuna for an easy way to do distributed optimization
    self.nb_alpha_params = self.nb_alpha.split(",")
    self.next(self.train, foreach="nb_alpha_params")
  

  @step
  def train(self):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Binarizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import cross_val_score
    from mlflow.models.signature import infer_signature
    import pickle
    import os

    self.alpha = float(self.input)
    print("Training NB with alpha %f" % self.alpha)
    self.clf = Pipeline([("bin", Binarizer()), ("nb", MultinomialNB(alpha=self.alpha))])
    self.clf.fit(self.train_vectors, self.y_train)
    self.score = cross_val_score(self.clf, self.train_vectors, self.y_train, scoring = "f1", cv = 10).mean()
    print(f"Alpha: {self.alpha}, CV F1 score: {self.score}")

    self.run_name = f"{current.flow_name}_{current.run_id}_{current.task_id}"
    with mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name) as run:
      mlflow.log_param("alpha", self.alpha)
      mlflow.log_metric("F1", self.score)
      signature = infer_signature(self.train_vectors, self.clf.predict(self.train_vectors))
      mlflow.sklearn.log_model(self.clf, "nb_classifier", signature=signature)
      vectorizer_path = f"vectorizer-{run.info.run_id}.pickle"
      with open(vectorizer_path, "wb") as handle:
        pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(vectorizer_path)
      os.remove(vectorizer_path)

    self.next(self.join)

  @step
  def join(self, inputs):
    from metaflow import Flow
    scores = [input.score for input in inputs]
    self.max_score = max(scores)
    max_index = scores.index(self.max_score)
    self.best_clf = inputs[max_index].clf
    # by accessing the vectorizer in the join because of the mlflow logging, can't include it in merge_artifacts
    self.vectorizer = inputs[max_index].vectorizer
    print("Max CV F1:", self.max_score, "Alpha:", self.best_clf["nb"].alpha)
    latest_run = Flow(current.flow_name).latest_successful_run
    if latest_run and "max_score" in latest_run.data:
      print("Previous successful run CV F1 score", latest_run.data.max_score)
      if self.max_score < latest_run.data.max_score:
        raise Exception("New model is not better than previous")
    self.merge_artifacts(inputs, include=["test_vectors", "y_test"])
    self.next(self.end)

  @card(type="blank")
  @step
  def end(self):
    from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
    predicted = self.best_clf.predict(self.test_vectors)
    print("F1 test score:", f1_score(self.y_test, predicted))
    cm = confusion_matrix(predicted, self.y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.best_clf.classes_)
    disp.plot()
    current.card.append(Image.from_matplotlib(disp.figure_))
    print("TrainFlow is done.")


if __name__ == "__main__":
  TrainFlow()
