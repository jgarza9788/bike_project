import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = './bike_project/Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

print(rides)