from torch.utils.data import DataLoader


def data_preprocessing(dataset, N, num_timesteps, batch_size = 1, shuffle = False):
  from torch.utils.data import DataLoader
  import numpy as np
  data_list_total = np.zeros((N, num_timesteps-3, 9))
  print("pre-processing data...")
  for n, data in enumerate(dataset):
    print("n:", n, "data shape:", dataset.shape)
        
    for i in range(4, 7, 1):
        # features = data[:4, i-4:i].T # 0,1,2,3th states (4,6)
        prev_4_reactor_temps = data[4, i-4:i] # (1,4)
        prev_4_F_ag = data[6, i-4:i] # (1,4)
        current_temp = data[4, i] # 4th state (1,)
        data_list_total[n, i-4, 0:4] = prev_4_reactor_temps
        data_list_total[n, i-4, 4:8] = prev_4_F_ag
        data_list_total[n, i-4, 8] = current_temp
    data_list_ready_for_model = DataLoader(data_list_total, batch_size=batch_size, shuffle=shuffle)
    return data_list_ready_for_model


def NeuralNet(num_hidden_layers, input_size = 24, hidden_size = 64, output_size = 1):
    import torch.nn as nn
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, output_size))
    model = nn.Sequential(*layers)
    return model

def train(model, train_loader, optimizer = "Adam", num_epochs = 1):
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam (model.parameters(), lr=0.01)
    losses = []
    j = 0
    for epoch in range (num_epochs):
        for n, data in enumerate (train_loader):
            print("n:", n, "data shape:", data.shape, "data.shape[1]:", data.shape[1])
            for i in range(data.shape[1]):
                # print("i:", i)

                prev_4_states = data[0, i, :-1].float().flatten()
                # print("prev_4_states size", prev_4_states.size())

                current_temp = data[0, i, -1].float().flatten()
                # print("current_state size", current_temp.size())
                
                predicted_current_temp = model(prev_4_states) # (1,)
                print("predicted_current_temp:", predicted_current_temp.item(), "ground_truth:", current_temp.item())

                loss = loss_func (current_temp, predicted_current_temp)
                losses.append (loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j+=1
                if j % 100 == 0:
                    print(f"Global iterations {j} | loss = {loss}")
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | loss = {loss}")
             
    plt.plot (losses)
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss");

def test(model, test_loader):
    """
    This function tests the model on the test set
    """
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt

    test_losses = []
    test_loss = 0
    loss_func = nn.MSELoss()
    model_output = []
    ground_truth = []
   
    with torch.no_grad():
        for n, data in enumerate (test_loader):
            print("n:", n, "data shape:", data.shape, "data.shape[1]:", data.shape[1])
            for i in range(data.shape[1]):
                # print("i:", i)

                prev_4_states = data[0, i, :-1].float().flatten()
                # print("prev_4_states size", prev_4_states.size())

                current_temp = data[0, i, -1].float().flatten()
                # print("current_state size", current_temp.size())
                
                predicted_current_temp = model(prev_4_states) # (1,)

                loss = loss_func (current_temp[-1].unsqueeze(dim=0), predicted_current_temp)
                model_output.append (predicted_current_temp.item())
                ground_truth.append (current_temp[0].item())
                print("predicted_current_temp:", predicted_current_temp.item(), "ground_truth:", current_temp[0].item())
                
                loss = loss_func (current_temp[0].item(), predicted_current_temp.item())
                test_losses.append (loss.item())

                test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    
    print("Total test loss:", test_loss)
    plt.plot (test_losses)
    plt.title("Loss Plot")

    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.show();

    # Plot model output and ground truth
    plt.plot (model_output, label = "Model Output")
    plt.plot (ground_truth, label = "Ground Truth")
    plt.title("Model Output vs Ground Truth")
    plt.xlabel("Timesteps")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.show()
    plt.xlabel("Epochs")
    plt.ylabel("Loss");

