import zenoh
import zenoh.handlers
import time


def particle_filter_update(lidar_data: str):
    """Your particle filter logic. Simulated here as a slow operation."""
    print(f"[PF] Processing latest: {lidar_data}")
    time.sleep(0.5)  # Simulate PF runtime


if __name__ == "__main__":
    with zenoh.open(zenoh.Config()) as session:
        # RingChannel(1) keeps ONLY the latest sample.
        # New arrivals overwrite the old one, so you never process stale backlogs.
        subscriber = session.declare_subscriber(
            "sensor/lidar", zenoh.handlers.RingChannel(1)
        )

        print("Subscriber started. Waiting for lidar data...")

        while True:
            # Blocks until a sample is available, but always returns the newest one.
            sample = subscriber.recv()

            payload = sample.payload.to_string()

            # Run your particle filter on the latest data only.
            particle_filter_update(payload)
