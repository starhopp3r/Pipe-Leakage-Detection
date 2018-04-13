"""
Initial script authored by Charles Lim. Modified by
Nikhil Raghavendra on 8/4/2018.
"""
import numpy as np
import random


# generate moisture sensor readings
def generate(ambience=100, noise=50, sensors=10, leaking=False):
    # create an array of baseline moisuture values using normal distribution
    readings = np.random.normal(loc=ambience, scale=noise, size=sensors)
    readings = np.absolute(readings)

    if leaking:
        leak_points = random.choice(range(1, sensors + 1))

        # we add 400 to the moisture level to the first few pipes
        # (will shuffle values later)
        for k in range(leak_points):
            leak_value = np.random.normal(loc=400, scale=noise, size=1)
            readings[k] += leak_value

    # sensors have min and max of 0 - 1024
    readings = np.clip(readings, 0, 1024)
    # sensors will return a discrete value,
    # so we make sure our test data is discrete too
    readings = readings.astype(int)
    np.random.shuffle(readings)
    return readings


samples = 10000
fake_dataset = np.zeros(shape=(samples, 11), dtype=int)


def sampleData():
    ambience = random.choice(range(50, 600))
    isLeaking = random.choice([True, False])
    readings = generate(ambience=ambience, leaking=isLeaking)
    return isLeaking, readings


for k in range(samples):
    isLeaking, readings = sampleData()
    readings = np.append(readings, [isLeaking])
    fake_dataset[k] = readings

np.savetxt("fake_data.csv", fake_dataset, fmt="%i", delimiter=",")
