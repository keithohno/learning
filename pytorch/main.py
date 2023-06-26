from mnist import run as run_mnist
from point_in_circle import run as run_point_in_circle
from point_in_cluster import run as run_point_in_cluster

print(
    "\nExamples:\n" "  (1) point_in_circle\n" "  (2) point_in_cluster\n" "  (3) mnist\n"
)

selection = input("Make a selection (default 3): ")

if selection == "1":
    run_point_in_circle()
elif selection == "2":
    run_point_in_cluster()
else:
    run_mnist()
