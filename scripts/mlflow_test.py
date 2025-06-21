import dagshub
dagshub.init(repo_owner='Pankaj220', repo_name='yt-comment-sentiment-analysis', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)