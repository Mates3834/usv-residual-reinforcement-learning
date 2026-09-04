import matplotlib.pyplot as plt

from experiments.evaluate_pid import evaluate_pid


trajectory, path, info = evaluate_pid()

plt.figure()
plt.plot(path[:, 0], path[:, 1], "--", label="Planned path")
plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    label="PID trajectory",
)

plt.scatter(path[0, 0], path[0, 1], marker="o", label="Start")
plt.scatter(path[-1, 0], path[-1, 1], marker="x", label="Goal")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Generic USV Navigation Demo")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

print("Evaluation:", info)
