import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("histogram_data.csv")

# Plot Histogram
plt.figure(figsize=(8, 5))
plt.plot(df["Bin"], df["Histogram"])
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram.png")

# Plot Cumulative Histogram
plt.figure(figsize=(8, 5))
plt.plot(df["Bin"], df["Cumulative"])
plt.title("Cumulative Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_histogram.png")

print("Saved histogram.png and cumulative_histogram.png")
