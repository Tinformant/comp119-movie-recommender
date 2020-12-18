import os, json
import argparse
import apache_beam as beam
from beam_nuggets.io import relational_db
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from pipeline_confi import (
    INPUT_SUBSCRIPTION,
    SERVICE_ACCOUNT_PATH,
    SOURCE_CONFIG_PROD,
    TABLE_CONFIG,)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH

parser = argparse.ArgumentParser()
path_args, pipeline_args = parser.parse_known_args()

##
# This is the main process which will
# lowercase data
# filter other type and only left movie type
##

def low(element):
    data = json.loads(element.decode("utf-8"))
    data["Plot"] = data["Plot"].lower()
    data["Actors"] = data["Actors"].lower()
    data["Director"] = data["Director"].lower()
    return data


def filter_movie(element):
    data = json.loads(element.decode("utf-8"))
    if data["Type"] != "movie":
        return False
    return True


def filter_out_nones(row):
  if row is not None:
    yield row

# This is the entry point to our pipeline
def run_main(pipeline_arguments):
    options = PipelineOptions(
        flags = pipeline_arguments,
        runner='DataflowRunner',
        project='big-data-292604',
        temp_location='gs://data_flow-movie-bucket/',
        region='us-central1')
    options.view_as(StandardOptions).streaming = True
    p = beam.Pipeline(options=options)  # initializing Pipeline object

    main_pipeline = (
        p
        | "Read data from pub sub"
        >> beam.io.ReadFromPubSub(subscription=INPUT_SUBSCRIPTION)
        | "Stripping newline character" >> beam.Map(lambda data: data.rstrip().lstrip())
        | "Filter other type only keep movie">> beam.filter(lambda data: filter_movie(data))
        | "Filter NaN data">> beam.filter(lambda data: filter_out_nones(data))
        | "lower" >> beam.Map(lambda data: low(data))
        | "Writing final data to production db"
        >> relational_db.Write(
            source_config=SOURCE_CONFIG_PROD, table_config=TABLE_CONFIG
        )
    )

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    run_main(pipeline_args)


#RUN PIPELINE ON GCP
# python mainpipeline.py --runner DataflowRunner \
# --project big-data-292604 \
# --temp_location gs://dataflow-movie-bucket/ \
# --requirements_file requirement.txt \
# --setup_file ./setup.py

