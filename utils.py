import matplotlib.pyplot as plt

def draw_plot(train, val, type):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation {}".format(type))
    plt.plot(val, label="val")
    plt.plot(train, label="train")
    plt.xlabel("Iterations")
    plt.ylabel(type)
    plt.legend()
    plt.show()