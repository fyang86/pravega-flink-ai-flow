#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
import os
import time
from typing import List

import ai_flow as af
import pandas as pd
from ai_flow import DatasetMeta
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage
from ai_flow_plugins.job_plugins import flink
from ai_flow_plugins.job_plugins.flink import FlinkPythonProcessor
from ai_flow_plugins.job_plugins.python.python_processor import ExecutionContext, PythonProcessor
from joblib import dump, load
from pyflink.table import Table, ScalarFunction, DataTypes, TableEnvironment
from pyflink.table.udf import udf
from sklearn.neighbors import KNeighborsClassifier

EXAMPLE_COLUMNS = ['sl', 'sw', 'pl', 'pw', 'type']
flink.set_flink_env(flink.FlinkStreamEnv())


class DatagenSource(FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        data_meta: DatasetMeta = execution_context.config['dataset']
        table_env: TableEnvironment = execution_context.table_env
        table_env.execute_sql('''
            create table {source_name} (
                sl FLOAT,
                sw FLOAT,
                pl FLOAT,
                pw FLOAT,
                type FLOAT
            ) with (
                'connector' = 'filesystem',
                'path' = '{uri}',
                'format' = 'csv',
                'csv.ignore-parse-errors' = 'true'
            )
        '''.format(uri=data_meta.uri, source_name=data_meta.name))
        table = table_env.from_path(data_meta.name)
        return [table]


class DatagenExecutor(FlinkPythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        return input_list


class DatagenSink(FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        data_meta: DatasetMeta = execution_context.config['dataset']
        sink, stream = 'datagen_' + data_meta.name.split('_')[0] + '_sink', data_meta.name.split('_')[0] + '-stream'
        table_env: TableEnvironment = execution_context.table_env
        statement_set = execution_context.statement_set
        table_env.execute_sql('''
                    create table {sink_name} (
                        sl FLOAT,
                        sw FLOAT,
                        pl FLOAT,
                        pw FLOAT,
                        type FLOAT
                    ) with (
                        'connector' = 'pravega',
                        'controller-uri' = 'tcp://localhost:9090',
                        'scope' = 'scope',
                        'sink.stream' = '{stream_name}',
                        'format' = 'json'
                    )
                '''.format(sink_name=sink, stream_name=stream))
        statement_set.add_insert(sink, input_list[0])
        return []


class TrainSource(FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        table_env: TableEnvironment = execution_context.table_env
        table_env.execute_sql('''
            create table train_source (
                sl FLOAT,
                sw FLOAT,
                pl FLOAT,
                pw FLOAT,
                type FLOAT
            ) with (
                'connector' = 'pravega',
                'controller-uri' = 'tcp://localhost:9090',
                'scope' = 'scope',
                'scan.execution.type' = 'streaming',
                'scan.streams' = 'train-stream',
                'format' = 'json'
            )
        ''')
        table = table_env.from_path('train_source')
        return [table]


class ModelTrainer(FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table]) -> List[Table]:
        """
        Train and save KNN model
        """
        tab = input_list[0]
        train_data = tab.to_pandas()
        y_train_data = train_data.pop(EXAMPLE_COLUMNS[4])
        x_train, y_train = train_data.values, y_train_data.values
        model_meta: af.ModelMeta = execution_context.config.get('model_info')
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(x_train, y_train)

        # Save model to local
        model_path = os.path.dirname(os.path.realpath(__file__)) + '/saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        model_path = model_path + '/' + model_timestamp
        dump(clf, model_path)
        af.register_model_version(model=model_meta, model_path=model_path)
        return []


class ValidateDatasetReader(PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        """
        Read test dataset
        """
        dataset_meta: af.DatasetMeta = execution_context.config.get('dataset')
        x_test = pd.read_csv(dataset_meta.uri, header=0, names=EXAMPLE_COLUMNS)
        y_test = x_test.pop(EXAMPLE_COLUMNS[4])
        return [[x_test, y_test]]


class ModelValidator(PythonProcessor):

    def __init__(self, artifact):
        super().__init__()
        self.artifact = artifact

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        """
        Validate and deploy model if necessary
        """
        current_model_meta: af.ModelMeta = execution_context.config.get('model_info')
        deployed_model_version = af.get_deployed_model_version(model_name=current_model_meta.name)
        new_model_meta = af.get_latest_generated_model_version(current_model_meta.name)
        uri = af.get_artifact_by_name(self.artifact).uri
        if deployed_model_version is None:
            # If there is no deployed model for now, update the current generated model to be deployed.
            af.update_model_version(model_name=current_model_meta.name,
                                    model_version=new_model_meta.version,
                                    current_stage=ModelVersionStage.VALIDATED)
            af.update_model_version(model_name=current_model_meta.name,
                                    model_version=new_model_meta.version,
                                    current_stage=ModelVersionStage.DEPLOYED)
        else:
            x_validate = input_list[0][0]
            y_validate = input_list[0][1]
            knn = load(new_model_meta.model_path)
            scores = knn.score(x_validate, y_validate)
            deployed_knn = load(deployed_model_version.model_path)
            deployed_scores = deployed_knn.score(x_validate, y_validate)

            with open(uri, 'a') as f:
                f.write(
                    'deployed model version: {} scores: {}\n'.format(deployed_model_version.version, deployed_scores))
                f.write('generated model version: {} scores: {}\n'.format(new_model_meta.version, scores))
            if scores >= deployed_scores:
                # Deprecate current model and deploy better new model
                af.update_model_version(model_name=current_model_meta.name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.DEPRECATED)
                af.update_model_version(model_name=current_model_meta.name,
                                        model_version=new_model_meta.version,
                                        current_stage=ModelVersionStage.VALIDATED)
                af.update_model_version(model_name=current_model_meta.name,
                                        model_version=new_model_meta.version,
                                        current_stage=ModelVersionStage.DEPLOYED)
        return []


class PredictSource(FlinkPythonProcessor):
    def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        """
        Flink source reader that reads from Pravega
        """
        t_env = execution_context.table_env
        t_env.execute_sql('''
            CREATE TABLE predict_source (
                sl FLOAT,
                sw FLOAT,
                pl FLOAT,
                pw FLOAT,
                type FLOAT
            ) WITH (
                'connector' = 'pravega',
                'controller-uri' = 'tcp://localhost:9090',
                'scope' = 'scope',
                'scan.execution.type' = 'batch',
                'scan.streams' = 'predict-stream',
                'format' = 'json'
            )
        ''')
        table = t_env.from_path('predict_source')
        return [table]


class Predictor(FlinkPythonProcessor):
    def __init__(self):
        super().__init__()
        self.model_name = None

    def open(self, execution_context: flink.ExecutionContext):
        self.model_name = execution_context.config['model_info'].name

    def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        """
        Use pyflink udf to do prediction
        """
        model_meta = af.get_deployed_model_version(self.model_name)
        model_path = model_meta.model_path
        clf = load(model_path)

        # Define the python udf

        class Predict(ScalarFunction):
            def eval(self, sl, sw, pl, pw):
                records = [[sl, sw, pl, pw]]
                df = pd.DataFrame.from_records(records, columns=['sl', 'sw', 'pl', 'pw'])
                return clf.predict(df)[0]

        # Register the udf in flink table env, so we can call it later in SQL statement
        execution_context.table_env.register_function('mypred',
                                                      udf(f=Predict(),
                                                          input_types=[DataTypes.FLOAT(), DataTypes.FLOAT(),
                                                                       DataTypes.FLOAT(), DataTypes.FLOAT()],
                                                          result_type=DataTypes.FLOAT()))
        return [input_list[0].select("mypred(sl,sw,pl,pw)")]


class PredictSink(FlinkPythonProcessor):

    def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        """
        Sink Flink Table produced by Predictor to local file
        """
        table_env = execution_context.table_env
        table_env.execute_sql('''
           CREATE TABLE predict_sink (
               prediction FLOAT 
           ) WITH (
               'connector' = 'filesystem',
               'path' = '{uri}',
               'format' = 'csv',
               'csv.ignore-parse-errors' = 'true'
           )
       '''.format(uri=execution_context.config['dataset'].uri))
        execution_context.statement_set.add_insert("predict_sink", input_list[0])
