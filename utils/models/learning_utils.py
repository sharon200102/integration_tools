def make_train_step(model, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # forward
        forward_dict = model(x, y)
        # Computes loss
        loss_dict = model.loss_function(forward_dict)
        total_loss = loss_dict['total_loss']
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss_dict

    # Returns the function that will be called inside the train loop
    return train_step
