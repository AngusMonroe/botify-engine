
import mongoengine
import config

def connect():
    mongoengine.connect(config.MONGO_DB_NAME, host = config.MONGO_HOST, port = config.MONGO_PORT)
