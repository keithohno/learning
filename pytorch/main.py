from classification_incircle import run as run0
from classification_clusters import run as run1
from classification_mnist import run as run2
from autoencoder_mnist import run as run3

print(
    "\nExamples:\n" "  (0) classification_circle\n" "  (1) classification_clusters\n" "  (2) classification_mnist\n" "  (3) autoencoder_mnist\n"
)

selection = input("Make a selection (default 3): ")

print()

if selection == "0":
    run0()
elif selection == "1":
    run1()
elif selection == "2":
    run2()
else:
    run3()
