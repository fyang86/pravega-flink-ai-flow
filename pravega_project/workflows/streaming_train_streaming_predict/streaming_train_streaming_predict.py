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

import ai_flow as af
from ai_flow.model_center.entity.model_version_stage import ModelVersionEventType
from ai_flow.workflow.status import Status

from streaming_train_streaming_predict_processor import ModelTrainer, ValidateDatasetReader, ModelValidator, \
    Predictor, DatagenSource, DatagenExecutor, \
    DatagenSink, TrainSource, PredictSource, PredictSink

DATASET_URI = os.path.abspath(os.path.join(__file__, "../../../")) + '/resources/iris_{}.csv'


def run_workflow():
    # Init project
    af.init_ai_flow_context()

    artifact_prefix = af.current_project_config().get_project_name() + "." + \
                      af.current_workflow_config().workflow_name + "."

    # Generating data
    with af.job_config("datagen"):
        # Write train data to Pravega
        datagen_train_dataset = af.register_dataset(name='datagen_train_source',
                                                    data_format='csv',
                                                    uri=DATASET_URI.format('train'))
        datagen_train_source = af.read_dataset(dataset_info=datagen_train_dataset,
                                               read_dataset_processor=DatagenSource())
        datagen_train_channel = af.user_define_operation(input=[datagen_train_source],
                                                         processor=DatagenExecutor(),
                                                         name='datagen_train')
        train_dataset = af.register_dataset(name='train_source',
                                            data_format='csv',
                                            uri='tcp://localhost:9090')
        af.write_dataset(input=datagen_train_channel,
                         dataset_info=train_dataset,
                         write_dataset_processor=DatagenSink())

        # Write predict data to Pravega
        datagen_predict_dataset = af.register_dataset(name='datagen_predict_source',
                                                      data_format='csv',
                                                      uri=DATASET_URI.format('test'))
        datagen_predict_source = af.read_dataset(dataset_info=datagen_predict_dataset,
                                                 read_dataset_processor=DatagenSource())
        datagen_predict_channel = af.user_define_operation(input=[datagen_predict_source],
                                                           processor=DatagenExecutor(),
                                                           name='datagen_predict')
        predict_dataset = af.register_dataset(name='predict_source',
                                              data_format='csv',
                                              uri='tcp://localhost:9090')
        af.write_dataset(input=datagen_predict_channel,
                         dataset_info=predict_dataset,
                         write_dataset_processor=DatagenSink())

    # Training of model
    with af.job_config('train'):
        train_source = af.read_dataset(dataset_info=train_dataset,
                                       read_dataset_processor=TrainSource())
        model_info = af.register_model(model_name=artifact_prefix + 'KNN',
                                       model_desc='KNN model')
        af.train(input=[train_source],
                 model_info=model_info,
                 training_processor=ModelTrainer())

    # Validation of model
    with af.job_config('validate'):
        # Read validation dataset
        validate_dataset = af.register_dataset(name=artifact_prefix + 'validate_dataset',
                                               uri=DATASET_URI.format('test'))
        # Validate model before it is used to predict
        validate_read_dataset = af.read_dataset(dataset_info=validate_dataset,
                                                read_dataset_processor=ValidateDatasetReader())
        validate_artifact_name = artifact_prefix + 'validate_artifact'
        af.register_artifact(name=validate_artifact_name,
                             uri=os.path.dirname(os.path.realpath(__file__)) + '/validate_result')
        af.model_validate(input=[validate_read_dataset],
                          model_info=model_info,
                          model_validation_processor=ModelValidator(validate_artifact_name))

    # Prediction(Inference) using flink
    with af.job_config('predict'):
        # Read test data and do prediction
        predict_source = af.read_dataset(dataset_info=predict_dataset,
                                         read_dataset_processor=PredictSource())
        predict_channel = af.predict(input=[predict_source],
                                     model_info=model_info,
                                     prediction_processor=Predictor())
        # Save prediction result
        write_dataset = af.register_dataset(name=artifact_prefix + 'write_dataset',
                                            uri=os.path.dirname(os.path.realpath(__file__)) + '/predict_result')
        af.write_dataset(input=predict_channel,
                         dataset_info=write_dataset,
                         write_dataset_processor=PredictSink())

    # Define relation graph connected by control edge: datagen -> train -> validate -> predict
    af.action_on_job_status(job_name='train', upstream_job_name='datagen', upstream_job_status=Status.FINISHED)
    af.action_on_model_version_event(job_name='validate',
                                     model_version_event_type=ModelVersionEventType.MODEL_GENERATED,
                                     model_name=model_info.name)
    af.action_on_model_version_event(job_name='predict',
                                     model_version_event_type=ModelVersionEventType.MODEL_VALIDATED,
                                     model_name=model_info.name)

    af.workflow_operation.stop_all_workflow_executions(af.current_workflow_config().workflow_name)
    # Submit workflow
    af.workflow_operation.submit_workflow(af.current_workflow_config().workflow_name)
    # Run workflow
    af.workflow_operation.start_new_workflow_execution(af.current_workflow_config().workflow_name)


if __name__ == '__main__':
    run_workflow()
