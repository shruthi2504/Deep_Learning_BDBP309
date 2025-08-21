#From the given paper we chose the 1000G dataset(smallest) and tried to implement a feed forward neural network
#the dataset has
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


# def data_preprocess():
#     file_path = "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt"
#     data = pd.read_csv(file_path, sep="\t", index_col=0)
#
#     # print(data.shape) #here we have 22722 rows(target and landmark genes) and 465 columns(462 samples with ID,gene symbol,chr,coord)
#     BGEDV2_LM_ID = 'bgedv2_GTEx_1000G_lm.txt'
#     BGEDV2_TG_ID = 'bgedv2_GTEx_1000G_tg.txt'
# data_preprocess()


def splitting_dataset():
    file = "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt"
    data = pd.read_csv(file, sep="\t")


    landmark_map = pd.read_csv("map_lm.txt", sep="\t", header=None, names=["Gene_Symbol", "Ensembl_ID", "Probe_ID"])
    target_map = pd.read_csv("map_tg.txt", sep="\t", header=None, names=["Gene_Symbol", "Ensembl_ID", "Probe_ID"])

    # Create a new column with base gene IDs (no version)
    data["GeneID_base"] = data["TargetID"].str.split('.').str[0]
    # Extract the IDs (2nd column)
    landmark_ids = landmark_map["Ensembl_ID"].tolist()
    target_ids = target_map["Ensembl_ID"].tolist()

    print("Number of landmark genes:", len(landmark_ids))
    print("Number of target genes:", len(target_ids))

    landmark_df = data[data["GeneID_base"].isin(landmark_ids)].copy()
    target_df = data[data["GeneID_base"].isin(target_ids)].copy()

    print("Landmark data shape:", landmark_df.shape)
    print("Target data shape:", target_df.shape)


    #Checking if there are multiple versions of the ID in the dataset
    # print(data["TargetID"].head(10))
    # print(landmark_ids[:10])
    # print(target_ids[:10])

    # for gid in landmark_ids[:10]:  # checking first 10 as an example
    #     matches = data[data["GeneID_base"] == gid]["TargetID"].tolist()
    #     if matches:
    #         print(f"{gid} has versions: {matches}")
    #     else:
    #         print(f"{gid} not found in data")
    #
    # Same for target IDs
    # for gid in target_ids[:10]:
    #     matches = data[data["GeneID_base"] == gid]["TargetID"].tolist()
    #     if matches:
    #         print(f"{gid} has versions: {matches}")
    #     else:
    #         print(f"{gid} not found in data")

    # def check_multiple_versions(gene_list, gene_type="Landmark"):
    #     print(f"\nChecking {gene_type} genes for multiple versions...")
    #     for gid in gene_list:
    #         matches = data[data["GeneID_base"] == gid]["TargetID"].tolist()
    #         if len(matches) > 1:
    #             print(f"{gid} has multiple versions: {matches}")
    #
    # check_multiple_versions(landmark_ids, "Landmark")
    # check_multiple_versions(target_ids, "Target")
    #No multiple versions so we are just removing the . and what follows

    #WORKING---------

    sample_cols = landmark_df.columns[4:-1]  # all sample columns
    landmark_matrix = landmark_df[sample_cols].astype(np.float32).to_numpy()
    target_matrix = target_df[sample_cols].astype(np.float32).to_numpy()

    # Stack landmarks + targets
    combined_matrix = np.vstack([landmark_matrix, target_matrix])

    # Save the combined matrix
    np.save("combined.npy", combined_matrix)

    # Save row info: [start_row_landmarks, end_row_landmarks, end_row_targets]
    row_info = np.array([0, landmark_matrix.shape[0], landmark_matrix.shape[0] + target_matrix.shape[0]])
    np.save("row_info.npy", row_info)

    print("Saved 'combined.npy' and 'row_info.npy'")
    print("Landmark shape:", landmark_matrix.shape)
    print("Target shape:", target_matrix.shape)

    #WORKING------
    row_info = np.load("row_info.npy")
    print(row_info) #flandmark from 0 to 942 and 943 to 10463 are target genes

    # print(landmark_df[sample_cols].applymap(lambda x: isinstance(x, str)).sum())

    # print(data.columns.get_loc("GeneID_base"))

    # landmark_matrix = np.load("landmark.npy")
    # target_matrix = np.load("target.npy")
    #
    # print(landmark_matrix.shape)
    # print(target_matrix.shape)# shape of the array
    # print(landmark_matrix[:5, :5])






def main():
    splitting_dataset()


if __name__ == "__main__":
    main()


















# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset, random_split
#
# # -----------------------------
# # 1. Load 1000G RNA-Seq data
# # -----------------------------
# file_path = "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt"
#
# # Load with pandas (assuming tab-delimited file)
# data = pd.read_csv(file_path, sep="\t", index_col=0)
#
# # Inspect shape
# print("Data shape:", data.shape)  # rows=genes, columns=samples?
#
# # -----------------------------
# # 2. Prepare input (X) and output (Y)
# # -----------------------------
# # Example: let's treat first 943 genes as input (landmark) and remaining as output (target)
# num_landmark = 943
# X_data = data.iloc[:num_landmark, :].T.values  # transpose: samples x genes
# Y_data = data.iloc[num_landmark:, :].T.values
#
# # -----------------------------
# # 3. Normalize per gene (Z-score)
# # -----------------------------
# X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)
# Y_data = (Y_data - Y_data.mean(axis=0)) / Y_data.std(axis=0)
#
#
# # -----------------------------
# # 4. Prepare DataLoader
# # -----------------------------
# def prepare_data(X_data, Y_data, batch_size=32, train_frac=0.8):
#     X = torch.tensor(X_data, dtype=torch.float32)
#     Y = torch.tensor(Y_data, dtype=torch.float32)
#     dataset = TensorDataset(X, Y)
#
#     train_size = int(train_frac * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader
#
#
# train_loader, test_loader = prepare_data(X_data, Y_data)
#
#
# # -----------------------------
# # 5. Build neural network
# # -----------------------------
# def build_model(input_size, output_size, hidden_sizes=[2048, 2048]):
#     layers = []
#     in_size = input_size
#     for h in hidden_sizes:
#         layers.append(nn.Linear(in_size, h))
#         layers.append(nn.ReLU())
#         in_size = h
#     layers.append(nn.Linear(in_size, output_size))
#     model = nn.Sequential(*layers)
#     return model
#
#
# model = build_model(input_size=X_data.shape[1], output_size=Y_data.shape[1])
#
# # -----------------------------
# # 6. Define loss and optimizer
# # -----------------------------
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
# # -----------------------------
# # 7. Training loop
# # -----------------------------
# def train_model(model, train_loader, loss_fn, optimizer, num_epochs=20):
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for x_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             y_pred = model(x_batch)
#             loss = loss_fn(y_pred, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() * x_batch.size(0)
#         avg_loss = total_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
#
#
# train_model(model, train_loader, loss_fn, optimizer)
#
#
# # -----------------------------
# # 8. Evaluation
# # -----------------------------
# def evaluate_model(model, test_loader, loss_fn):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for x_batch, y_batch in test_loader:
#             y_pred = model(x_batch)
#             loss = loss_fn(y_pred, y_batch)
#             total_loss += loss.item() * x_batch.size(0)
#     avg_loss = total_loss / len(test_loader.dataset)
#     print(f"Test Loss: {avg_loss:.4f}")
#
#
# evaluate_model(model, test_loader, loss_fn)
