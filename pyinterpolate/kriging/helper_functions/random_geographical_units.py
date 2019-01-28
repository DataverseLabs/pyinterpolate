import numpy as np
import matplotlib.pyplot as plt


class RandomGeographicalUnits:
    """Class generates random areas from the point matrix"""

    def __init__(self, width=100, height=100, number_of_areas=8):

        # Base data frame
        self.generated_data = {}

        # Base matrix
        self.mtx = np.zeros(shape=(height, width))

        # Initial points
        initial_points_xs = np.random.randint(0, width, size=number_of_areas)
        initial_points_ys = np.random.randint(0, height, size=number_of_areas)
        self.initial_points = np.column_stack((initial_points_xs, initial_points_ys))

        # Create maps - division map - A
        self.divided_area = self.divide_area()
        # Create maps - population map - B
        self.population_area = self.create_population_area(height, width)
        # Create maps - counts map - C
        self.areas_counts = np.random.randint(0, 100, size=number_of_areas)
        self.counted_areas, self.rates, self.centroids = self.create_counted_areas()

    # A
    def divide_area(self):
        area = self.mtx.copy()
        for idx_row, row in enumerate(area):
            for idx_col, col in enumerate(row):
                if area[idx_row, idx_col] == 0:
                    area[idx_row, idx_col] = self.calculate_manhattan([idx_col, idx_row])
        print('Area division units created successfully')
        return area

    # B
    @staticmethod
    def create_population_area(height, width):
        area = np.random.poisson(1, size=(height, width)) * np.abs(np.random.normal(loc=10, size=(height, width)))
        return area

    # C
    def create_counted_areas(self):
        areas = self.divided_area.copy()
        rates = self.divided_area.copy()
        centroids = []
        for idx, val in enumerate(self.areas_counts):
            coordinates = np.where(self.divided_area == idx + 1)
            coordinates = np.column_stack(coordinates)
            centroid = np.mean(coordinates, axis=0)
            centroid = [centroid[0], centroid[1]]
            centroids.append(centroid)
            areas[self.divided_area == idx + 1] = val
            population_at_area = int(np.sum(self.population_area[self.divided_area == idx + 1]))
            r = (val / population_at_area) * 100000
            rates[self.divided_area == idx + 1] = r

            # Update data dictionary
            self.generated_data[idx + 1] = []
            self.generated_data[idx + 1].append({'coordinates': coordinates})
            self.generated_data[idx + 1].append({'centroid': centroid})
            self.generated_data[idx + 1].append({'population': population_at_area})
            self.generated_data[idx + 1].append({'cases': val})
            self.generated_data[idx + 1].append({'rates': r})

        return areas, rates, np.asarray(centroids)

    # Additional functions
    def calculate_manhattan(self, point):
        point_value = 0
        distance = 10 ** 6
        points = self.initial_points.copy()
        for idx, pt in enumerate(points):
            dist = np.abs(pt[0] - point[0]) + np.abs(pt[1] - point[1])
            if dist < distance:
                distance = dist
                point_value = idx + 1
        return point_value

    # Data visualization methods
    def show_area(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.divided_area)
        plt.scatter(self.centroids[:, 1], self.centroids[:, 0], c='black')
        plt.title('Generated areas')
        plt.plot()

    def show_population(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.population_area)
        plt.title('Population data')
        plt.colorbar()
        plt.plot()

    def show_areas_values(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.counted_areas)
        plt.title('Areas values')
        plt.colorbar()
        plt.plot()

    def show_areas_rates(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.rates)
        plt.title('Areas rates')
        plt.colorbar()
        plt.plot()
