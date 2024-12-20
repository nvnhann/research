"""Producer base-class providing common utilites and functionality"""
import logging
import time


from confluent_kafka import avro
from confluent_kafka.admin import AdminClient, NewTopic
from confluent_kafka.avro import AvroProducer

logger = logging.getLogger(__name__)


class Producer:
    """Defines and provides common functionality amongst Producers"""

    # Tracks existing topics across all Producer instances
    existing_topics = set([])

    def __init__(
        self,
        topic_name,
        key_schema,
        value_schema=None,
        num_partitions=1,
        num_replicas=1,
    ):
        """Initializes a Producer object with basic settings"""
        self.topic_name = topic_name
        self.key_schema = key_schema
        self.value_schema = value_schema
        self.num_partitions = num_partitions
        self.num_replicas = num_replicas
        self.broker_properties = {
        "schema.registry.url": "http://localhost:8081",
        "bootstrap.servers": "PLAINTEXT://localhost:9092"
        }

        # If the topic does not already exist, try to create it
        if self.topic_name not in Producer.existing_topics:
            self.create_topic()
            Producer.existing_topics.add(self.topic_name)

        #  Configure the AvroProducer
        self.producer = AvroProducer(
            {
            "bootstrap.servers": "PLAINTEXT://localhost:9092",
            "schema.registry.url": "http://localhost:8081"},
            default_key_schema=key_schema,
            default_value_schema=value_schema,
        )

    def create_topic(self):
        """Creates the producer topic if it does not already exist"""
        adminclient = AdminClient({"bootstrap.servers": self.broker_properties["bootstrap.servers"]})
        list_topic = adminclient.list_topics()
        if self.topic_name in list_topic.topics:
            logger.info(f'Topic: {self.topic_name} already exist')
            return
        topics = adminclient.create_topics([
                            NewTopic(
                                topic=self.topic_name, 
                                num_partitions= self.num_partitions, 
#                                 replication_factor=self.num_replicas,
                                replication_factor=1
                            )
                            ])
        for t, r in topics.items():
            try:
                r.result()
                logger.info("Topic %s is created!!", t)
            except Exception as ex:
                logger.error("Error(%s): %s!", t, ex)

    def close(self):
        """Prepares the producer for exit by cleaning up the producer"""
        if self.producer is not None:
            self.producer.flush()
        logger.info("producer close incomplete - skipping")

    def time_millis(self):
        """Use this function to get the key for Kafka Events"""
        return int(round(time.time() * 1000))
