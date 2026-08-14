import zenoh
import random
import time

if __name__ == "__main__":
    # Create a Zenoh session
    session = zenoh.open(zenoh.Config())

    # Create a publisher for the topic "test/topic"
    publisher = session.declare_publisher("test/topic")

    try:
        while True:
            # Generate a random integer between 1 and 100
            random_value = random.randint(1, 100)

            # Publish the random value to the topic
            publisher.put(f"{random_value}")

            print(f"Published: {random_value}")

            # Sleep for 1 second before publishing the next value
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping publisher...")
    finally:
        # Close the Zenoh session
        session.close()
