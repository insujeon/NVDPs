import matplotlib.pyplot as plt

# Visualization
def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points. 

    Args: 
        target_x: An array of shape [B, num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B, num_targets,1] that contains the 
            y values of the target points.
        context_x: An array of shape [B, num_contexts, 1] that contains
            the x values of the context points.
        context_y: An array of shape [B, num_contexts, 1] that contains
            the y values of the context points.
        pred_y: An array of shape [B, num_targets, 1] that contains the 
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B, num_targets, 1] that contains the 
            predicted std dev of the y values at the target points in target_x. 
    """

    # Plot everythong
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
            target_x[0, :, 0],
            pred_y[0, :, 0] - std[0, :, 0],
            pred_y[0, :, 0] + std[0, :, 0],
            alpha=0.25,
            facecolor='green',
            interpolate=True)

    # Make the ploot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2,2])
    plt.grid('off')
    ax = plt.gca()



# Visualization
def plot_functions_fixed(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points. 

    Args: 
        target_x: An array of shape [B, num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B, num_targets,1] that contains the 
            y values of the target points.
        context_x: An array of shape [B, num_contexts, 1] that contains
            the x values of the context points.
        context_y: An array of shape [B, num_contexts, 1] that contains
            the y values of the context points.
        pred_y: An array of shape [B, num_targets, 1] that contains the 
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B, num_targets, 1] that contains the 
            predicted std dev of the y values at the target points in target_x. 
    """

    # Plot everythong
    for s in range(10):
        plt.plot(target_x[0], pred_y[0,s,:,:], 'b', linewidth=2)

    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)

    # plt.fill_between(
    #         target_x[0, :, 0],
    #         pred_y[0, :, 0] - std[0, :, 0],
    #         pred_y[0, :, 0] + std[0, :, 0],
    #         alpha=0.25,
    #         facecolor='green',
    #         interpolate=True)

    # Make the ploot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2,2])
    plt.grid('off')
    ax = plt.gca()
