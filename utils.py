import matplotlib.pyplot as plt

def draw_plot(train, val, type, save_name=None):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation {}".format(type))
    plt.plot(val, label="val")
    plt.plot(train, label="train")
    plt.xlabel("Iterations")
    plt.ylabel(type)
    plt.legend()
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')