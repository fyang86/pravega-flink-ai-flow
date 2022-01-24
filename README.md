# PravegaFlinkAIFlow
A sample using Pravega as external source for Flink AI Flow

## Install Flink AI Flow
Before installing Flink AI Flow, please make sure the python version is 3.7.x and MySQL Client is installed.
You can check [AI Flow prerequisites](https://ai-flow.readthedocs.io/en/latest/content/deployment/installation.html#prerequisites)
for more details to install these prerequisites.

Use pip to install AI Flow 0.2.2:
```
pip install ai-flow==0.2.2
```

## Deploy Flink cluster
1. Download and unzip Flink 1.11.4 cluster: https://flink.apache.org/downloads.html
2. Download Pravega flink connectors dependency from https://repo1.maven.org/maven2/io/pravega/pravega-connectors-flink-1.11_2.12/0.10.1/
    and move it to flink `/lib` directory
3. Change the flink `conf/flink-conf.yaml`, set `taskmanager.numberOfTaskSlots: ` from 1 to 4 to allow more parallelism
4. Start Flink cluster, you can see the Flink Web UI in `localhost:8081` to check it's running fine:
```
./bin/start-cluster.sh
```

## Deploy Pravega cluster
1. Download and unzip Pravega 0.10.1: https://github.com/pravega/pravega/releases/tag/v0.10.1
2. Start Pravega standalone cluster:
```
./bin/pravega-standalone
```

## Deploy Flink AI Flow
1. Prepare your AIFlow server Configuration with the following example in `$HOME/aiflow/aiflow_server.yaml`:
```yaml
# Config of AIFlow server

# port of AIFlow server
server_port: 50051
# uri of database backend for AIFlow server
db_uri: sqlite:///${AIFLOW_HOME}/aiflow.db
# type of database backend for AIFlow server, can be SQL_LITE, MYSQL, MONGODB
db_type: SQL_LITE

# uri of the server of notification service
notification_server_uri: 127.0.0.1:50052

# whether to start the metadata service, default is True
#start_meta_service: True

# whether to start the model center service, default is True
#start_model_center_service: True

# whether to start the metric service, default is True
#start_metric_service: True

# whether to start the scheduler service, default is True
#start_scheduler_service: True

# scheduler config
scheduler_service:
  scheduler:
    scheduler_class: ai_flow_plugins.scheduler_plugins.airflow.airflow_scheduler.AirFlowScheduler
    scheduler_config:
      # AirFlow dag file deployment directory, i.e., where the airflow dag will be. If it is not set, the dags_folder in
      # airflow config will be used
      #airflow_deploy_path: /tmp/dags

      # Notification service uri used by the AirFlowScheduler.
      notification_server_uri: 127.0.0.1:50052
  # The path to a local directory where the scheduler service download the Workflow codes.
  #repository: /tmp

# web server config
web_server:
  airflow_web_server_uri: http://localhost:8080
  host: 0.0.0.0
  port: 18000
```
If there's already config file in `$HOME/aiflow/aiflow_server.yaml`, you need to mannually change 
AI Flow webserver default port from 8000 to 18000 to avoid port conflict with Pravega.
2. Start all AI Flow services(AI Flow, Airflow and Notification service):
```
start-all-aiflow-services.sh
```


## Run the demo
First we need to create sample scope and stream in Pravega. Open Pravega cli: `./pravega-0.10.1/bin/pravega-cli`:
```
scope create scope
stream create scope/train-stream
stream create scope/predict-stream
```


```
cd $HOME
git clone https://github.com/fyang86/pravega-flink-ai-flow
cd pravega-flink-ai-flow/pravega_project/workflows/batch_train_batch_predict
python batch_train_batch_predict.py
```
You can check [AIFlow Web](localhost:18000) with the default username(admin) and password(admin) to see the workflow metadata, and the graph of the workflow
and [Apache Airflow](localhost:8080) with the default username(admin) and password(admin) to view the execution of workflows.

After the predict job succeed, you can check the predict result in `pravega-flink-ai-flow/pravega_project/workflows/batch_train_batch_predict/predict_result`.

Finally, run following command to stop all services:
```
stop-all-aiflow-services.sh
```