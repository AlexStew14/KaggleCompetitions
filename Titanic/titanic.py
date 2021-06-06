import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


train_data = pd.read_csv(r"./data/train.csv")
age_and_survived = train_data[["Age", "Survived"]].dropna()

age = np.array(age_and_survived["Age"])
survived_col = np.array(age_and_survived["Survived"])

age_died = age[survived_col == 0]
age_survived = age[survived_col == 1]

print(age_survived)


n_bins = 25
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(age_survived, bins=n_bins)
axs[1].hist(age_died, bins=n_bins)
plt.show()

x = age.reshape(-1, 1)
y = survived_col

clf = LogisticRegression(random_state=0).fit(x, y)

print(clf.score(x, y))
