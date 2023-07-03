print(
    "\nExamples:\n"
    "  (0) classification_circle\n"
    "  (1) classification_clusters\n"
    "  (2) classification_mnist\n"
    "  (3) autoencoder_mnist\n"
)

selection = input("Make a selection (default 3): ")

print()

if selection == "0":
    import classification_incircle

    classification_incircle.run()

elif selection == "1":
    import classification_clusters

    classification_clusters.run()

elif selection == "2":
    import classification_mnist

    classification_mnist.run()

else:
    import autoencoder_mnist

    autoencoder_mnist.run()
