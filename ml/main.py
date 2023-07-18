print(
    "\nExamples:\n"
    "  (0) classification_circle\n"
    "  (1) classification_clusters\n"
    "  (2) classification_mnist\n"
    "  (3) autoencoder_mnist\n"
    "  (4) vae_mnist\n"
)

selection = input("Make a selection (default 4): ")

print()

if selection == "0":
    from examples import classification_incircle

    classification_incircle.run()

elif selection == "1":
    from examples import classification_clusters

    classification_clusters.run()

elif selection == "2":
    from examples import classification_mnist

    classification_mnist.run()

elif selection == "3":
    from examples import autoencoder_mnist

    autoencoder_mnist.run()

else:
    from examples import vae_mnist

    vae_mnist.run()
