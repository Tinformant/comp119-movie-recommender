from beam_nuggets.io import relational_db

##
# Change the subscription details as per your GCP project
##
INPUT_SUBSCRIPTION = "projects/big-data-292604/subscriptions/moviesubscription"

##
# Change the path to the dir where your service account private key file is kept
##
SERVICE_ACCOUNT_PATH = "./cre.json"


##
# Change the details as per your MYSQL config
##
SOURCE_CONFIG_PROD = relational_db.SourceConfiguration(
    drivername="mysql+pymysql",
    host="XXX.XXX.XXX.XX",
    port=3306,
    username="movie",
    password="bigdata",
    database="movie_recommendation",
    create_if_missing=False,  # create the database if not there
)

##
# Change the details as per your table name
##
TABLE_CONFIG = relational_db.TableConfiguration(
    name="movie_data",
    create_if_missing=True,  # automatically create the table if not there
    primary_key_columns=["ImdbID"],  # and use 'num' column as primary key
)