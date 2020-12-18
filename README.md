# comp119-movie-recommender

Tufts University COMP119 Big Data Final Project

Team: Runze Si, Run Yu and Yujia He

**recommender.py** contains code for the movie recommendation algorithms. Three different algorithms, demographic, content-based and collaborative filtering.

**pubsub.py** contains the code for requesting new data from API and broadcasting them with Pub/Sub.

**pipeline_config.py** is the script for configuring Pub/Sub and MySql.

**mainpipeline.py** is the entry point for different runners (local, Dataflow, etc) for running the pipeline. In this pipeline script, we are reading data from the Pub/Sub，process and filter data and and storing the final data in a relational database.


