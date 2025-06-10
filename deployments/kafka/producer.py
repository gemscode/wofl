from kafka import KafkaProducer
import os
import time

bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
topic = os.environ.get("SETUP_TOPIC", "videos-views")

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
for i in range(5):
    msg = f"Sample message {i}".encode("utf-8")
    producer.send(topic, msg)
    print(f"Sent: {msg.decode()}")
    time.sleep(1)

producer.flush()
print("Successfully produced sample messages")

