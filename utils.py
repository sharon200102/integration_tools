def make_train_step(model,optimizer):
  # Builds function that performs a step in the train loop
  def train_step(x,y):
    # Sets model to TRAIN mode
    model.train()
    # forward
    forward_dict = model(x,y)
    # Computes loss
    loss = model.loss_function(forward_dict)
    # Computes gradients
    loss.backward()
    # Updates parameters and zeroes gradients
    optimizer.step()
    optimizer.zero_grad()
    # Returns the loss
    return loss.item()

  # Returns the function that will be called inside the train loop
  return train_step