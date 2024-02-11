import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

#temporary single file opening
file = open("T-Data-PC/T-NG-NM-1.json")
json_read = file.read()
# storing full dictionary
raw_dict = json.loads(json_read)

def selectIndices(raw_dict, frame, close_loop = False):
    """
    Construct a new list of X and Y values of each eyelid landmark for a given frame
 
    :param frame: The indexed frame
    :param raw_dict: The current dictionary being read
    :param close_loop: Boolean flag indicating whether to append first index to end for animation
    :return: List of tuples containing X and Y values for each selected landmark index
    """ 
    indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    selected_indices = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in indices]

    if close_loop:
        selected_indices.append(raw_dict['apps']['frames'][frame]['mp'][33][:2])

    return selected_indices

#application of Green's theorum. Sum of cross products around each vertex to calculate area
def polyArea(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

#get amount of frames
num_frames = len(raw_dict['apps']['frames'])

start_time = time.time()
#calculate average area
sum_area = 0
for i in range(num_frames):
    list = selectIndices(raw_dict, i)
    sum_area += polyArea(list)
average_area = sum_area / num_frames

#compare current area to average area
sum_area = 0
blink_count = 0
blink_indices = []
for i in range(num_frames):
    list = selectIndices(raw_dict, i)
    current_area = polyArea(list)
    if current_area < average_area - 0.005:
        blink_count += 1
        blink_indices.append(i)

end_time = time.time()


print("Selected indices: ",blink_indices)
print("Amount of blinks:",blink_count)
print("Elapsed Time:", (end_time - start_time) * 1000, "ms")
