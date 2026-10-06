import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Create Delivery Time Dataset
# --------------------------------------------------

np.random.seed(42)

# Generate 97 normal delivery times
delivery_times = np.random.normal(
    loc=32,
    scale=7,
    size=97
)

# Convert values into realistic whole minutes
delivery_times = np.round(delivery_times).astype(int)

# Add 3 unusually long delivery times
delivery_times = np.append(
    delivery_times,
    [68, 71, 75]
)


# Create Order IDs
order_ids = np.arange(1, 101)


# Create DataFrame
data = {
    "OrderID": order_ids,
    "DeliveryTime(min)": delivery_times
}

delivery_df = pd.DataFrame(data)


# Save dataset to CSV
delivery_df.to_csv(
    "delivery_times.csv",
    index=False
)


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = pd.read_csv("delivery_times.csv")


print("Dataset Shape:", df.shape)


# --------------------------------------------------
# 3. Mean and Standard Deviation
# --------------------------------------------------

real_mean = df["DeliveryTime(min)"].mean()

real_std = df["DeliveryTime(min)"].std()


print(
    f"\nReal delivery times -- "
    f"Mean: {real_mean:.2f} min "
    f"Std: {real_std:.2f} min"
)


# --------------------------------------------------
# 4. Simulate Expected Delivery Times
# --------------------------------------------------

simulated_times = np.random.normal(
    loc=real_mean,
    scale=real_std,
    size=100
)


print(
    f"Simulated 100 values from "
    f"Normal(mean={real_mean:.2f}, "
    f"std={real_std:.2f})"
)


# --------------------------------------------------
# 5. Compare Real and Simulated Distributions
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=12,
    alpha=0.6,
    label="Real Delivery Times",
    edgecolor="black"
)

plt.hist(
    simulated_times,
    bins=12,
    alpha=0.6,
    label="Simulated Delivery Times",
    edgecolor="black"
)

plt.title("Real vs Simulated Delivery Times")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    "delivery_comparison_histogram.png"
)

plt.show()


# --------------------------------------------------
# 6. Missing Values
# --------------------------------------------------

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)


# --------------------------------------------------
# 7. Boxplot
# --------------------------------------------------

plt.figure(figsize=(6, 4))

plt.boxplot(
    df["DeliveryTime(min)"]
)

plt.title("Delivery Time Boxplot")
plt.ylabel("Delivery Time (minutes)")

plt.tight_layout()

plt.savefig("delivery_boxplot.png")

plt.show()


# --------------------------------------------------
# 8. Calculate Percentiles
# --------------------------------------------------

q1 = df["DeliveryTime(min)"].quantile(0.25)

median = df["DeliveryTime(min)"].quantile(0.50)

q3 = df["DeliveryTime(min)"].quantile(0.75)


print(
    f"\nPercentiles -- "
    f"25th: {q1:.2f} "
    f"50th: {median:.2f} "
    f"75th: {q3:.2f}"
)


# --------------------------------------------------
# 9. IQR Outlier Detection
# --------------------------------------------------

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr

upper_bound = q3 + 1.5 * iqr


print(
    f"IQR outlier bounds: "
    f"{lower_bound:.2f} - "
    f"{upper_bound:.2f}"
)


# Find outliers
outliers = df[
    (df["DeliveryTime(min)"] < lower_bound)
    |
    (df["DeliveryTime(min)"] > upper_bound)
]


print(
    "\nOutliers detected:",
    len(outliers)
)


print(
    outliers[
        ["OrderID", "DeliveryTime(min)"]
    ]
)


# --------------------------------------------------
# 10. Percentage of Outliers
# --------------------------------------------------

outlier_percentage = (
    len(outliers) / len(df)
) * 100


print(
    f"\nPercentage flagged as outliers: "
    f"{outlier_percentage:.1f}%"
)


print(
    "\nSaved "
    "delivery_comparison_histogram.png "
    "and delivery_boxplot.png"
)