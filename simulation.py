import simpy
import random
import pandas as pd
import numpy as np

def bank_simulation(arrival_rate, service_rate, num_servers, simulation_time):
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=num_servers)

    waiting_times = []
    queue_lengths = []

    def customer(env):
        arrival_time = env.now

        with server.request() as request:
            queue_lengths.append(len(server.queue))
            yield request

            waiting_time = env.now - arrival_time
            waiting_times.append(waiting_time)

            service_time = random.expovariate(service_rate)
            yield env.timeout(service_time)

    def customer_arrivals(env):
        while True:
            yield env.timeout(random.expovariate(arrival_rate))
            env.process(customer(env))

    env.process(customer_arrivals(env))
    env.run(until=simulation_time)

    avg_wait = np.mean(waiting_times) if waiting_times else 0
    max_queue = max(queue_lengths) if queue_lengths else 0

    return avg_wait, max_queue


# Generate 1000 simulations
data = []

for i in range(1000):
    arrival_rate = random.uniform(0.5, 2.0)
    service_rate = random.uniform(0.5, 2.5)
    num_servers = random.randint(1, 5)
    simulation_time = random.randint(100, 500)

    avg_wait, max_queue = bank_simulation(
        arrival_rate,
        service_rate,
        num_servers,
        simulation_time
    )

    data.append([
        arrival_rate,
        service_rate,
        num_servers,
        simulation_time,
        avg_wait,
        max_queue
    ])

df = pd.DataFrame(data, columns=[
    "arrival_rate",
    "service_rate",
    "num_servers",
    "simulation_time",
    "avg_waiting_time",
    "max_queue_length"
])

df.to_csv("simulation_data.csv", index=False)

print("✅ 1000 Enhanced Simulations Generated Successfully!")