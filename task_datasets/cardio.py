# cardio_train.csv
import os
import numpy as np
import pandas as pd


class CardioDataset:
    def __init__(self, train=True, transform=None, download=False, root=os.environ.get("DATAROOT"), split=None):
        super(CardioDataset, self).__init__()
        
        self.data = pd.read_csv(os.path.join(root, "cardio_train.csv"), sep=";")
        X_raw = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.data = pd.DataFrame(data, columns=self.data.columns)
        # self.data = self.df_transform()
        if split is not None:
            if split == "train":
                self.data = self.data[:int(len(self.data) * 0.8)]
            elif split == "val":
                self.data = self.data[int(len(self.data) * 0.8) : int(len(self.data) * 0.9)]
            elif split == "test":
                self.data = self.data[int(len(self.data) * 0.9):]
        else:
            if train:
                self.data = self.data[:int(len(self.data) * 0.9)]
            else:
                self.data = self.data[int(len(self.data) * 0.9):]
        self.transform = transform
        
    def df_transform(self):
        # age: int
        # gender: one_hot
        # height: int
        # weight: int
        # ap_hi: int
        # ap_lo: int
        # cholesterol: one_hot
        # gluc: one_hot
        # smoke: one_hot
        # alco: one_hot
        # active: one_hot
        df = self.data
        # normalize age to 0-1
        # df["age"] = df["age"] / (df["age"].max() - df["age"].min())
        # The line `df["age"] = df["age"] / (df["age"].max() - df["age"].min())` in the `df_transform`
        age = df["age"] / (df["age"].max() - df["age"].min())
        gender = pd.get_dummies(df["gender"]).astype(int)
        height = df["height"] / (df["height"].max() - df["height"].min())
        weight = df["weight"] / (df["weight"].max() - df["weight"].min())
        ap_hi = df["ap_hi"] / (df["ap_hi"].max() - df["ap_hi"].min())
        ap_lo = df["ap_lo"] / (df["ap_lo"].max() - df["ap_lo"].min())
        cholesterol = pd.get_dummies(df["cholesterol"]).astype(int)
        gluc = pd.get_dummies(df["gluc"]).astype(int)
        smoke = pd.get_dummies(df["smoke"]).astype(int)
        alco = pd.get_dummies(df["alco"]).astype(int)
        active = pd.get_dummies(df["active"]).astype(int)
        X = pd.concat([
            age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        ], axis=1)
        y = df["cardio"]
        data = pd.concat([X, y], axis=1)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, :-1].values
        y = self.data.iloc[idx, -1]
        return x, y


def feature_transform(x):
    # Add your feature transformation logic here
    # Modify the input x and return the transformed x
    # check type of x
    # if 
    # ID number

    # age
    # in days

    # gender
    # 1 - women, 2 - men

    # height
    # cm

    # weight
    # kg

    # ap_hi
    # Systolic blood pressure

    # ap_lo
    # Diastolic blood pressure

    # cholesterol
    # 1: normal, 2: above normal, 3: well above normal

    # gluc
    # 1: normal, 2: above normal, 3: well above normal

    # smoke
    # whether patient smokes or not

    # alco
    # Binary feature

    # active
    # Binary feature

    # cardio
    # Target variable
    pass

if __name__ == "__main__":
    train_dataset = CardioDataset(train=True, transform=None)
    test_dataset = CardioDataset(train=False, transform=None)
    print(train_dataset[0][0], train_dataset[0][1])
    
