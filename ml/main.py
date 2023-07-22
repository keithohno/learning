print(
    "\nExamples:\n"
    "  (0) classification_circle\n"
    "  (1) classification_clusters\n"
    "  (2) classification_mnist\n"
    "  (3) autoencoder_mnist\n"
    "  (4) vae_mnist\n"
    "  (5) vae_fashion\n"
    "  (6) gan_mnist\n"
)

selection = input("Make a selection (default 6): ")

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

elif selection == "4":
    from examples import vae_mnist

    vae_mnist.run()

elif selection == "5":
    from examples import vae_fashion

    vae_fashion.run()


elif selection == "6":
    from examples import gan_mnist

    gan_mnist.run()

else:
    from examples import gan_mnist

    gan_mnist.run()
