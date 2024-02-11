import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

def main():
    #temporary single file opening 
    file = open("T-Data-PC/T-NG-NM-1.json")
    json_read = file.read()
    # storing full dictionary
    raw_dict = json.loads(json_read)

    num_frames = len(raw_dict['apps']['frames'])

    global slope
    global intercept

    slope = -0.000023060176766189095
    intercept = 0.0014811993621631767

    start_time = time.time()
    blink_indices, blink_count, average_area = detect_blinks(raw_dict, num_frames, selectIndices, calculate_triangle_area, calculate_average_area, eye_choice_input="right", THRESHOLD=0.5)
    end_time = time.time()

    print(f"Blink count: {blink_count}")
    print(f'Average Area: {average_area:.8f}')
    print(f"Elapsed Time: {(end_time - start_time) * 1000} ms")
    print(f"Frames registered as blinks: {blink_indices}")

    print(len(distances))
    slope, intercept = np.polyfit(distances, eye_areas, 1)
    print(slope, intercept)

    # Use the model to predict eye areas at given distan  m ces
    predicted_eye_areas = slope * distances + intercept

    # Plotting
    plt.scatter(distances, eye_areas, label='Measured Data')
    plt.plot(distances, predicted_eye_areas, label='Model Fit', color='red')
    plt.xlabel('Distance')
    plt.ylabel('Eye Area')
    plt.legend()
    plt.show()

    generate_eye_animation(selectIndices, normalize_eye_area, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=70, slow_down_multiplier=1)

"""
Construct a new list of X and Y values of each eyelid landmark for a given frame

:param close_loop: Boolean flag indicating whether to append first index to end for drawing animation
:param eye_choice: String 'left' or 'right' eye
:return: array of vertices for each eyelid landmark
"""       

EYE_INDICES_LEFT = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
EYE_INDICES_RIGHT = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

def selectIndices(raw_dict, frame, close_loop = False, only_iris = False, eye_choice = "left"):

    eye_indices = EYE_INDICES_RIGHT if eye_choice == "right" else EYE_INDICES_LEFT

    # if close_loop, append first vertex to end of array so animation polygon is complete
    if close_loop and eye_choice == "left":
        selected_vertices_left = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_left.append(raw_dict['apps']['frames'][frame]['mp'][263][:2])
        return np.array(selected_vertices_left)
    elif close_loop and eye_choice == "right":
        selected_vertices_right = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_right.append(raw_dict['apps']['frames'][frame]['mp'][33][:2])
        return np.array(selected_vertices_right)

    if only_iris and eye_choice == "left":
        iris_indices_left = [472, 471, 470, 469]
        iris_vertices_left = [raw_dict['apps']['frames'][frame]['mp'][index][0] for index in iris_indices_left]
        return np.array(iris_vertices_left)
     
    selected_vertices = [raw_dict['apps']['frames'][frame]['mp'][index][:3] for index in eye_indices]


    return np.array(selected_vertices)

def calculate_triangle_area(v1, v2, v3):
    vec1 = v2 - v1
    vec2 = v3 - v1

    cross_prod = np.cross(vec1, vec2)

    return np.linalg.norm(cross_prod) / 2

def calculate_average_area(raw_dict, num_frames, selectIndices, calculate_triangle_area, eye_choice_input):

    all_frames_total_area = 0

    global average_distance

    distances_sum = 0
    for i in range(num_frames):
        width = 1280

        irisMinX = -1
        irisMaxX = -1

        vertices = selectIndices(raw_dict, i, only_iris = True)

        for selectedX in vertices:  # Directly iterate over x-coordinates
            if irisMinX == -1 or selectedX * width < irisMinX:
                irisMinX = selectedX * width
            if irisMaxX == -1 or selectedX * width > irisMaxX:
                irisMaxX = selectedX * width
                
        dx = irisMaxX - irisMinX
        dX = 11.7

        fx = 664

        dZ = (fx * (dX / dx))  / 10.0
        distances_sum += dZ
    average_distance = distances_sum / num_frames

    for i in range(num_frames):
        width = 1280

        irisMinX = -1
        irisMaxX = -1

        vertices = selectIndices(raw_dict, i, only_iris = True)

        for selectedX in vertices:  # Directly iterate over x-coordinates
            if irisMinX == -1 or selectedX * width < irisMinX:
                irisMinX = selectedX * width
            if irisMaxX == -1 or selectedX * width > irisMaxX:
                irisMaxX = selectedX * width
                
        dx = irisMaxX - irisMinX
        dX = 11.7

        fx = 664

        dZ = (fx * (dX / dx))  / 10.0

        total_area = 0
        vertices = selectIndices(raw_dict, i, eye_choice = eye_choice_input)
        triangles = np.array([
            [0,1,15],[1,15,14],[1,2,14],[2,14,13],[2,13,3],[3,13,12],[3,12,4],
            [12,4,11],[4,11,5],[11,5,10],[5,10,6],[6,10,9],[6,9,7],[7,9,8]
        ])
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            total_area += calculate_triangle_area(v1, v2, v3)
        all_frames_total_area += normalize_eye_area(total_area, dZ, slope, intercept, average_distance)
    average_area = all_frames_total_area / num_frames
    return average_area, average_distance

def normalize_eye_area(measured_area, measured_distance, m, b , D_ref):
    A_ref = m * D_ref + b

    A_expected_area = m * measured_distance + b
    
    normalized_area = measured_area * (A_ref / A_expected_area)

    return normalized_area

"""
Detect Blinks

:param THRESHOLD: percentage of average area that must be reached for a given frames area to register as a blink
"""
def detect_blinks(raw_dict, num_frames, selectIndices, calculate_triangle_area, calculate_average_area, THRESHOLD = 0.551, eye_choice_input = "both"):    
    blink_count = 0
    blink_indices = []
    start, end = None, None
    if eye_choice_input == "both": 
        average_area, average_distance = (calculate_average_area(raw_dict, num_frames, selectIndices, calculate_triangle_area, "left") + calculate_average_area(raw_dict, num_frames, selectIndices, calculate_triangle_area, "right")) / 2
    else:
        average_area, average_distance = calculate_average_area(raw_dict, num_frames, selectIndices, calculate_triangle_area, eye_choice_input)

    global distances
    global eye_areas

    distances = np.array([])
    eye_areas = np.array([])

    for i in range(num_frames):

        width = 1280

        irisMinX = -1
        irisMaxX = -1

        vertices = selectIndices(raw_dict, i, only_iris = True)

        for selectedX in vertices:  # Directly iterate over x-coordinates
            if irisMinX == -1 or selectedX * width < irisMinX:
                irisMinX = selectedX * width
            if irisMaxX == -1 or selectedX * width > irisMaxX:
                irisMaxX = selectedX * width

        dx = irisMaxX - irisMinX
        dX = 11.7

        fx = 664

        dZ = (fx * (dX / dx))  / 10.0

        current_area = 0
        vertices = selectIndices(raw_dict, i, eye_choice = eye_choice_input)
        triangles = np.array([
            [0,1,15],[1,15,14],[1,2,14],[2,14,13],[2,13,3],[3,13,12],[3,12,4],
            [12,4,11],[4,11,5],[11,5,10],[5,10,6],[6,10,9],[6,9,7],[7,9,8]
        ])
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            current_area += calculate_triangle_area(v1, v2, v3)

        current_area = normalize_eye_area(current_area, dZ, slope, intercept, average_distance)

        distances = np.append(distances, dZ)
        eye_areas = np.append(eye_areas, current_area)

        if current_area < average_area * THRESHOLD:
            if start is None:
                start = i
            end = i
        else:
            if start is not None:
                blink_indices.append((start, end))
                blink_count += 1
                start, end = None, None
    if start is not None:
        blink_indices.append((start, end))
    return blink_indices, blink_count, average_area

"""
Draw MatPlotLib animation.
:param graph_eye_choice: String 'left', 'right', or 'both' eyes.
:param interval: duration of each frame in milliseconds
:param slow_down_multiplier: how many times to repeat blink frames (slows down animation when blink event is met)
"""
def generate_eye_animation(selectIndices, normalize_eye_area, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=50, slow_down_multiplier=4):

    frames_list_left = [selectIndices(raw_dict, frame, close_loop=True, eye_choice="left") for frame in range(num_frames)]
    frames_list_right = [selectIndices(raw_dict, frame, close_loop=True, eye_choice="right") for frame in range(num_frames)]
        
    #set scale for plot
    if graph_eye_choice == "left":
        x_values = [point[0] for frame in frames_list_left for point in frame]
        y_values = [point[1] for frame in frames_list_left for point in frame]
    elif graph_eye_choice == "right":
        x_values = [point[0] for frame in frames_list_right for point in frame]
        y_values = [point[1] for frame in frames_list_right for point in frame]
    elif graph_eye_choice == "both":
        # Combine x values from both eyes
        x_values_left = [point[0] for frame in frames_list_left for point in frame]
        x_values_right = [point[0] for frame in frames_list_right for point in frame]
        x_values = x_values_left + x_values_right

        # Combine y values from both eyes
        y_values_left = [point[1] for frame in frames_list_left for point in frame]
        y_values_right = [point[1] for frame in frames_list_right for point in frame]
        y_values = y_values_left + y_values_right
    else:
        raise ValueError("Invalid graph_eye_choice. Must be 'left', 'right', or 'both'.")

    # Calculate minimum and maximum for x and y values
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Create figure and axis with fixed limits
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Initialize empty plot and text element
    if graph_eye_choice == "both":
        line_left, = ax.plot([], [], 'bo-')
        line_right, = ax.plot([], [], 'bo-')
    else:
        line, = ax.plot([], [], 'bo-')

    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    blink_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    blink_true = ax.text(0.1, 0.9, '', transform=ax.transAxes, color = "green")
    blink_false = ax.text(0.1, 0.9, '', transform=ax.transAxes, color = "red")
    blink_count_text = ax.text(0.2, 0.95, '', transform=ax.transAxes)
    current_area_text = ax.text(0.2, 0.9, '', transform=ax.transAxes)
    distance_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

    # Function to initialize plot
    def init():
        frame_text.set_text('')
        blink_text.set_text('')
        distance_text.set_text('')
        if graph_eye_choice == "both":
            line_left.set_data([], [])
            line_right.set_data([], [])
            return line_left, line_right, frame_text
        else:
            line.set_data([], [])
            return line, frame_text

    # Variable for slowing down on registered blinks
    current_frame = [0]

    # Function to update the plot for each frame
    def update(frame_index):

        is_blink = any(start <= current_frame[0] <= end for start, end in blink_indices)

        if is_blink:
            blink_true.set_text('True')
            blink_false.set_text('')
            if frame_index % slow_down_multiplier == 0:
                current_frame[0] += 1
        else:
            blink_true.set_text('')
            blink_false.set_text('False')
            current_frame[0] += 1

        frame = current_frame[0]  # This is now used to index into your frames_list_left/right

        # Ensure frame does not exceed available data
        frame = min(frame, len(frames_list_left) - 1, len(frames_list_right) - 1)

        # Calculate current area to display in graph
        current_area = 0
        vertices = selectIndices(raw_dict, frame)
        triangles = np.array([
            [0,1,15],[1,15,14],[1,2,14],[2,14,13],[2,13,3],[3,13,12],[3,12,4],
            [12,4,11],[4,11,5],[11,5,10],[5,10,6],[6,10,9],[6,9,7],[7,9,8]
        ])
        for triangle in triangles:
            v1, v2, v3 = vertices[triangle]
            current_area += calculate_triangle_area(v1, v2, v3)

        width = 1280
        height = 720

        irisMinX = -1
        irisMaxX = -1

        vertices = selectIndices(raw_dict, frame_index, only_iris = True)

        for selectedX in vertices:  # Directly iterate over x-coordinates
            if irisMinX == -1 or selectedX * width < irisMinX:
                irisMinX = selectedX * width
            if irisMaxX == -1 or selectedX * width > irisMaxX:
                irisMaxX = selectedX * width

                
        dx = irisMaxX - irisMinX
        dX = 11.7

        fx = 664

        dZ = (fx * (dX / dx))  / 10.0

        distance_text.set_text(f'Distance: {dZ}cm')

        current_area = normalize_eye_area(current_area, dZ, slope, intercept, average_distance)

        #logic to draw vertices
        if graph_eye_choice == "both":
            vertices_left = frames_list_left[frame]
            vertices_right = frames_list_right[frame]
            x_left, y_left = zip(*vertices_left)
            x_right, y_right = zip(*vertices_right)
            line_left.set_data(x_left, y_left)
            line_right.set_data(x_right, y_right)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_area * 1000:.5f}')
            return line_left, line_right, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text, distance_text
        elif graph_eye_choice == "left":
            vertices = frames_list_left[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_area * 1000:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text, distance_text
        elif graph_eye_choice == "right":
            vertices = frames_list_right[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_area * 1000:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text, distance_text

    # Calculate total number of frames including repeated frames for slow down
    extra_frames_for_blinks = sum((end - start + 1) * (slow_down_multiplier - 1) for start, end in blink_indices)
    num_frames = len(frames_list_left) + extra_frames_for_blinks

    animation = FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=True, interval=interval, repeat=False)
        
    plt.show()

if __name__ == "__main__":
    main()