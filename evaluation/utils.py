import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(training_loss, validation_loss):
    epoch_loss = [sum(epoch)/len(epoch) for epoch in training_loss]
    epoch_loss_val = [sum(epoch)/len(epoch) for epoch in validation_loss]

    plt.plot(range(1, len(epoch_loss)+1), epoch_loss, label="Training")
    plt.plot(range(1, len(epoch_loss)+1), epoch_loss_val, label="Validation")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.tight_layout
    plt.show()
    

def plot_training_curve(loss, smoothing):
    loss_flat = [batch for epoch in loss for batch in epoch]
    epoch_mean_batch = np.cumsum([len(epoch) for epoch in loss])
    epoch_loss = [sum(epoch)/len(epoch) for epoch in loss]

    N = smoothing
    cumsum, moving_aves = [0], []
    for i, x in enumerate(loss_flat, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)

    plt.plot(moving_aves,zorder=1)
    plt.scatter(epoch_mean_batch, epoch_loss, s=5, c='r',zorder=2)
    plt.ylabel("Cross-entropy loss")
    plt.xticks(epoch_mean_batch, range(1, len(loss)+1))
    plt.xlabel("Epoch")
    plt.tight_layout
    plt.show()
    