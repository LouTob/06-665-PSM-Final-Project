from torch.utils.data import DataLoader


def data_preprocessing(dataset, N, num_timesteps, batch_size = 1, shuffle = False):
    from torch.utils.data import DataLoader
    import numpy as np
    data_list_total = np.zeros((N, num_timesteps-3, 9))
    print("pre-processing data with N =", N)
    for n in range(0, N, 1):
        # print("n:", n, "of", N, "dataset shape:", dataset.shape)
        i = 0    
        for i in range(4, 500, 1):
            # features = data[:4, i-4:i].T # 0,1,2,3th states (4,6)
            prev_4_reactor_temps = dataset[n, 4, i-4:i] # (1,4)
            prev_4_F_ag = dataset[n, 6, i-4:i] # (1,4)
            current_temp = dataset[n, 4, i] # 4th state (1,)
            # if i > 495:
            #     print("i:", i, "prev_4_reactor_temps:", prev_4_reactor_temps, "prev_4_F_ag:", prev_4_F_ag, "current_temp:", current_temp)
            # if i == 4:
            #     print("i:", i, "prev_4_reactor_temps:", prev_4_reactor_temps, "prev_4_F_ag:", prev_4_F_ag, "current_temp:", current_temp)
            data_list_total[n, i-4, 0:4] = prev_4_reactor_temps
            data_list_total[n, i-4, 4:8] = prev_4_F_ag
            data_list_total[n, i-4, 8] = current_temp
            # if i > 495:
            #     print("data_list_total[n, i-4, :]:", data_list_total[n, i-4, :])
    data_list_ready_for_model = DataLoader(data_list_total, batch_size=batch_size, shuffle=shuffle)
    return data_list_ready_for_model


def NeuralNet(num_hidden_layers, input_size = 24, hidden_size = 64, output_size = 1, dropout_rate = 0.0):
    import torch.nn as nn
    import torch
    # this method enables us to add as many layers as num_hidden_layers
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(nn.ReLU())
    for i in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(hidden_size, output_size))
    model = nn.Sequential(*layers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def train_and_test(model, train_loader, test_loader, optimizer = "Adam", num_epochs = 1):
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam (model.parameters(), lr=0.01)
    epoch_train_loss_list_mean_per_simulation = []
    overall_train_loss_list_one_value_per_epoch = []
    j = 0
    for epoch in range (num_epochs):
        model.train()
        print("Train epoch:", epoch)
        for n, data in enumerate (train_loader):
            # print("n:", n, "data shape:", data.shape, "data.shape[1]:", data.shape[1])
            train_loss_list_per_simulation = []
            for i in range(data.shape[1]-1):
                # print("i:", i)

                prev_4_states = data[0, i, :-1].float().flatten().to(device)
                # print("prev_4_states size", prev_4_states.size())

                current_temp = data[0, i, -1].float().flatten().to(device)
                # print("current_state size", current_temp.size())
                
                predicted_current_temp = model(prev_4_states) # (1,)
                loss = loss_func (current_temp, predicted_current_temp)
                train_loss_list_per_simulation.append (loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                j+=1
                # if j % 1000 == 0:
                    # print(f"Global iterations {j} | loss = {loss}")
            epoch_train_loss_list_mean_per_simulation.append(np.mean(train_loss_list_per_simulation))
        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Epoch mean training loss for a single simulation = {np.mean(epoch_train_loss_list_mean_per_simulation)}")
        overall_train_loss_list_one_value_per_epoch.append(np.mean(epoch_train_loss_list_mean_per_simulation))
        epoch_test_loss_list_mean_per_simulation = []
        overall_test_loss_list_one_value_per_epoch = []

        model = model.eval()
        print("Test epoch:", epoch)
        test_loss = 0
        with torch.no_grad():
            for n, data in enumerate (test_loader):
                # print("n:", n, "data shape:", data.shape, "data.shape[1]:", data.shape[1])
                model_output = []
                ground_truth = []
                test_loss_list_per_simulation = []
                for i in range(data.shape[1]-1):
                    # print("i:", i)

                    prev_4_states = data[0, i, :-1].float().flatten().to(device)
                    # print("prev_4_states size", prev_4_states.size())

                    current_temp = data[0, i, -1].float().flatten().to(device)
                    # print("current_state size", current_temp.size())
                    
                    predicted_current_temp = model(prev_4_states) # (1,)

                    model_output.append (predicted_current_temp.item()-273.15)
                    ground_truth.append (current_temp[0].item()-273.15)
                    # print("predicted_current_temp:", predicted_current_temp.item(), "ground_truth:", current_temp[0].item())
                    
                    loss = loss_func (current_temp, predicted_current_temp)
                    test_loss_list_per_simulation.append(loss.item())

                    test_loss += loss.item()
                epoch_test_loss_list_mean_per_simulation.append(np.mean(test_loss_list_per_simulation))

        print(f"Epoch {epoch} | Epoch mean test loss for a single simulation = {np.mean(epoch_test_loss_list_mean_per_simulation)}")
        overall_test_loss_list_one_value_per_epoch.append(np.mean(epoch_test_loss_list_mean_per_simulation))

        # Generate one plot sample for each epoch
        pred_error = []
        for i in range(len(model_output)):
            pred_error.append(ground_truth[i] - model_output[i])
        plt.plot(pred_error)
        plt.title(f"Prediction error (ground truth - prediction) for set n = {n}")
        plt.xlabel("Timesteps")
        plt.ylabel("Error")
        plt.show();

        # Plot model output and ground truth
        plt.plot(ground_truth, label = "Ground Truth")
        plt.plot(model_output, label = "Model Output")
        plt.title(f"Model Output vs Ground Truth for set n = {n}")
        plt.xlabel("Timesteps")
        plt.ylabel("Temperature (C)")
        plt.legend()
        plt.show();
      
        
      
    # Plot loss
    plt.plot(overall_train_loss_list_one_value_per_epoch, label = "Train Loss")
    plt.plot (overall_test_loss_list_one_value_per_epoch, label = "Test Loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show();         
        
        # plt.plot (train_losses)
        # plt.title("Loss Plot")
        # plt.yscale("log")   
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show(); 

        # plt.plot (train_losses)
        # plt.title("Loss Plot")
        # plt.yscale("log") 
        # plt.xscale("log")  
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show(); 
    return
def test(model, test_loader):
    """
    This function tests the model on the test set
    """
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_losses = []
    test_loss = 0
    loss_func = nn.MSELoss()
    model_output = []
    ground_truth = []
    model = model.eval()
    model = model.to(device)
    with torch.no_grad():
        for n, data in enumerate (test_loader):
            # print("n:", n, "data shape:", data.shape, "data.shape[1]:", data.shape[1])
            for i in range(data.shape[1]-1):
                # print("i:", i)

                prev_4_states = data[0, i, :-1].float().flatten().to(device)
                # print("prev_4_states size", prev_4_states.size())

                current_temp = data[0, i, -1].float().flatten().to(device)
                # print("current_state size", current_temp.size())
                
                predicted_current_temp = model(prev_4_states) # (1,)

                model_output.append (predicted_current_temp.item()-273.15)
                ground_truth.append (current_temp[0].item()-273.15)
                # print("predicted_current_temp:", predicted_current_temp.item(), "ground_truth:", current_temp[0].item())
                
                loss = loss_func (current_temp, predicted_current_temp)
                test_losses.append (loss.item())

                test_loss += loss.item()
            test_loss = test_loss / len(test_loader)
            
            print("Total test loss for set n =", n, "is", test_loss)

            pred_error = []
            for i in range(len(model_output)):
                pred_error.append(ground_truth[i] - model_output[i])
            plt.plot(pred_error)
            plt.title(f"Prediction error (ground truth - prediction) for set n = {n}")
            plt.xlabel("Timesteps")
            plt.ylabel("Error")
            plt.show();

            # Plot model output and ground truth
            plt.plot(ground_truth, label = "Ground Truth")
            plt.plot(model_output, label = "Model Output")
            plt.title(f"Model Output vs Ground Truth for set n = {n}")
            plt.xlabel("Timesteps")
            plt.ylabel("Temperature (C)")
            plt.legend()
            plt.show();

