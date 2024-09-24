import matplotlib.pyplot as plt
import numpy as np

def show_attention_batch(
        attention,
        xlabel='Keys',
        ylabel='Queries',
        title='Attention weights',
        figsize=(5, 5),
        cmap='Reds'
    ):
    """
    visualize attention weights in batch

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (b, m, n)
    """

    # Get the batch size
    batch_size, _, _ = attention.shape

    ncols = int(np.sqrt(batch_size))
    nrows = int(np.ceil(batch_size / ncols))

    # Create subplots for each attention matrix in the batch
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # Iterate through each attention matrix in the batch
    for i in range(nrows):
        for j in range(ncols):
            # Get the attention matrix
            ax = axs[i, j]
            ax.imshow(
                attention[i * ncols + j],
                cmap=cmap
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} - Batch {i * ncols + j + 1}')

    plt.tight_layout()
    plt.show()