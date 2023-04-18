from torch.utils.data import DataLoader


def data_preprocessing(dataset, batch_size = 1, shuffle = False):
  from torch.utils.data import DataLoader
  data_list = []
  for i, data in enumerate(dataset):
    features = data[:, i:i+4].T # 0,1,2,3th states (4,6)
    label = data[:, i+4] # 4th states (6,)
    data_list.append ((features,label))
  data_list_ready_for_model = DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
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

def train(model, train_loader, optimizer = "Adam", num_epochs = 100):
    import torch.nn as nn
    import torch
    import matplotlib.pyplot as plt
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam (model.parameters(), lr=0.01)
    num_epochs = 5
    losses = []

    for epoch in range (num_epochs):
        for i, (past, current) in enumerate (train_loader):
            input = past.float().flatten() # (24,)
            current = current.float().flatten() #(6,)
            # print("current shape:", current.shape, current.type())
            
            out = model(input) # (1,)

            # print("out shape:", out.shape, out.type(), out.item())
            # print("current-1 shape:", current[-1].unsqueeze(dim=0).shape, current[-1].type(), current[-1].item())

            loss = loss_func (current[-1].unsqueeze(dim=0), out)
            losses.append (loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print (f"iter {i} | loss = {loss}")
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
        for i, (past, current) in enumerate (test_loader):
            print(i)
            input = past.float().flatten() # (24,)
            current = current.float().flatten()
            # print("current:", current)
            # print("current-1:", current[-1])
            out = model(input)
            # print("out shape:", out.shape, out.type(), out.item())
            # print("current shape:", current[-1].shape, current.type(), current[-1].item())
            model_output.append (out.item())
            ground_truth.append (current[-1].item())
            loss = loss_func (current[-1].unsqueeze(dim=0), out)
            test_losses.append (loss.item())

            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    
    print("Total test loss:", test_loss)
    plt.plot (test_losses)
    plt.title("Loss Plot")
<<<<<<<< HEAD:NNPreprocessingAndStructure.py
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
    plt.show();
========
    plt.xlabel("Epochs")
    plt.ylabel("Loss");
>>>>>>>> e18467be2604fdbba0702460b956d71418f32dd6:python/NN preprocessing and structure.py
