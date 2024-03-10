import numpy as np
import matplotlib.pyplot as plt

# Function to load city coordinates from a text file
def load_city_coords(filename):
    with open(filename, 'r') as file:
        coords = []
        for line in file:
            x, y = map(float, line.strip().split())
            coords.append([x, y])
    return np.array(coords)

def ask_user_for_dataset_choice():
    print("What dataset do you want to run it on?")
    print("1: file-stp.txt")
    print("2: bays29file-tsp.txt")
    choice = input("Choose (1 or 2): ")
    return choice

choice = ask_user_for_dataset_choice()

if choice == "1":
    filename = 'file-tsp.txt'
elif choice == "2":
    filename = 'bays29file-tsp.txt'
else:
    print("Invalid choice")
    exit()
 
city_coords = load_city_coords(filename)


# This can be changed to the city order you want to visualize
city_order = [18, 27, 20, 12, 24, 1, 13, 4, 6, 3, 23, 10, 25, 2, 21, 5, 19, 11, 26, 17, 7, 22, 8, 15, 14, 9, 0, 28, 16]



# Plot the cities as points
plt.figure(figsize=(10, 8))
plt.plot(city_coords[:, 0], city_coords[:, 1], 'o', markerfacecolor='blue', markeredgecolor='black')

# Draw lines to show the path between cities
for i in range(-1, len(city_order) - 1):
    plt.plot([city_coords[city_order[i], 0], city_coords[city_order[i + 1], 0]],
             [city_coords[city_order[i], 1], city_coords[city_order[i + 1], 1]], 'k-')

# Annotates the cities with their order numbers
for index, (x, y) in enumerate(city_coords):
    plt.text(x, y, str(index), color="red", fontsize=8)

plt.title('Traveling Salesman Path from start')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.show()
