import matplotlib.pyplot as plt

def init_plot():
    plt.ion()  # Turn on the interactive mode
    plt.figure(figsize=(6, 10))  # Adjust figure size for vertical layout

def update_training_plot(epoch, train_loss, train_accuracy, test_loss, test_accuracy, refresh_rate=1):
    """
    Updates the training and test loss and accuracy plots for each epoch in a vertical layout.
    Refreshes the plot every 'refresh_rate' epochs.

    Parameters:
    - epoch: The current epoch number.
    - train_loss: The training loss for the current epoch.
    - train_accuracy: The training accuracy for the current epoch.
    - test_loss: The test loss for the current epoch.
    - test_accuracy: The test accuracy for the current epoch.
    - refresh_rate: The number of epochs between plot refreshes.
    """
    # Append the current epoch metrics to the lists
    if epoch == 0:
        global train_losses, train_accuracies, test_losses, test_accuracies
        train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Only update the plot every 'refresh_rate' epochs
    if epoch % refresh_rate == 0:
        plt.clf()  # Clear the current figure
        # Plotting training and test losses
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'ro-', label='Training Loss')
        plt.plot(range(1, len(test_losses) + 1), test_losses, 'bo-', label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting training and test accuracies
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'ro-', label='Training Accuracy')
        plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'bo-', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.pause(0.1)  # Pause for a short period to update the plot

def finish_plot():
    plt.ioff()  # Turn off the interactive mode
    plt.savefig("training progress.png")
    plt.show()