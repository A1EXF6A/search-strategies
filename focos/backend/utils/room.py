import numpy
from numpy.typing import NDArray

LIGHTBULBS: NDArray = numpy.array(
    [
        (2, 2),
        (4, 2),
        (6, 2),
        (2, 5),
        (4, 5),
        (6, 5),
    ],
    dtype=numpy.float32,
)

POINTS_COUNT: int = 42

LIGHTBULB_POWER: float = 15
TIME: float = 5

COST_PER_KWH: float = 0.106


class Room:
    points: NDArray

    uniform_weight: float

    consumed_energy: float = LIGHTBULB_POWER * TIME / 1000

    cost_per_lightbulb: float = consumed_energy * COST_PER_KWH

    def __init__(self, uniform_weight: float):
        self.uniform_weight = uniform_weight

        self.points = numpy.zeros(
            (POINTS_COUNT, LIGHTBULBS.shape[0]), dtype=numpy.float32
        )

        self.calculate_points()

        # for i in range(POINTS_COUNT):
        #     print(f"Point ({(i // 6) + 1}, {(i % 6) + 1}): {self.points[i]}")

    def calculate_points(self) -> None:
        for i in range(1, 8):
            for j in range(1, 7):
                for k in range(LIGHTBULBS.shape[0]):
                    point: tuple[float, float] = (i, j)
                    objetive: tuple[float, float] = LIGHTBULBS[k]

                    index: int = (i - 1) * 6 + (j - 1)

                    value: float = 1 / (
                        1
                        + (objetive[0] - point[0]) ** 2
                        + (objetive[1] - point[1]) ** 2
                    )

                    self.points[index][k] = value

    def lighting(self, individual: NDArray) -> float:
        avg_lighting: float = 0.0
        min_lighting: float = self.points[0][0]

        for i in range(POINTS_COUNT):
            lighting_at_point: float = 0.0

            for j in range(LIGHTBULBS.shape[0]):
                if individual[j] == 0:
                    continue

                lighting_at_point += self.points[i][j]

            min_lighting = min(min_lighting, lighting_at_point)

            lighting_at_point = min(lighting_at_point, 1.0)

            avg_lighting += lighting_at_point

        avg_lighting /= POINTS_COUNT

        room_lighting: float = avg_lighting - self.uniform_weight * (
            avg_lighting - min_lighting
        )

        return room_lighting

    def cost(self, individual: NDArray) -> float:
        lightbulbs_on: int = numpy.sum(individual)

        return lightbulbs_on * self.cost_per_lightbulb

    def metrics(self, individual: NDArray) -> dict[str, float]:
        avg_lighting: float = 0.0
        min_lighting: float = self.points[0][0]

        for i in range(POINTS_COUNT):
            lighting_at_point: float = 0.0

            for j in range(LIGHTBULBS.shape[0]):
                if individual[j] == 0:
                    continue

                lighting_at_point += self.points[i][j]

            min_lighting = min(min_lighting, lighting_at_point)

            lighting_at_point = min(lighting_at_point, 1.0)

            avg_lighting += lighting_at_point

        avg_lighting /= POINTS_COUNT

        lighting: float = avg_lighting - self.uniform_weight * (
            avg_lighting - min_lighting
        )

        uniformity: float
        if avg_lighting > 0:
            uniformity = min_lighting / avg_lighting
        else:
            uniformity = 0.0

        return {
            "avg_lighting": avg_lighting,
            "min_lighting": min_lighting,
            "lighting": lighting,
            "uniformity": uniformity,
            "cost": self.cost(individual),
            "consumed_energy": self.consumed_energy,
            "cost_per_lightbulb": self.cost_per_lightbulb,
        }
