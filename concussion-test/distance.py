
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import math

def main():
    #temporary single file opening 
    file = open("T-Data-PC/T-G-Hor-3.json")
    json_read = file.read()
    # storing full dictionary
    raw_dict = json.loads(json_read)

    num_frames = len(raw_dict['apps']['frames'])

    start_time = time.time()
    blink_indices, blink_count = detect_blinks(raw_dict, selectIndices, num_frames)
    end_time = time.time()

    print(blink_indices)
    print(blink_count)
    print("Elapsed Time:", (end_time - start_time) * 1000, "ms")

    generate_eye_animation(selectIndices, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=25, slow_down_multiplier=4)

"""
Construct a new list of X and Y values of each eyelid landmark for a given frame

:param close_loop: Boolean flag indicating whether to append first index to end for drawing animation
:param eye_choice: String 'left' or 'right' eye
:return: array of vertices for each eyelid landmark
""" 
def selectIndices(raw_dict, frame, close_loop = False, eye_choice = "left"):

    if eye_choice == "left":
        eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    elif eye_choice == "right":
        eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    else:
        print("Invalid eye choice")
        return
    
    eye_indices_calculation = [159,145]

    selected_vertices = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices_calculation]

    if close_loop and eye_choice == "left":
        selected_vertices_left = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_left.append(raw_dict['apps']['frames'][frame]['mp'][263][:2])
        return np.array(selected_vertices_left)
        pass
    elif close_loop and eye_choice == "right":
        selected_vertices_right = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_right.append(raw_dict['apps']['frames'][frame]['mp'][33][:2])
        return np.array(selected_vertices_right)

    return np.array(selected_vertices)

def detect_blinks(raw_dict, selectIndices, num_frames):
    all_frames_total_distance = 0

    for i in range(num_frames):
        values = selectIndices(raw_dict,i)
        x1, y1 = values[0]
        x2, y2 = values[1]
        distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        all_frames_total_distance += distance
    average_distance = all_frames_total_distance / num_frames

    blink_indices = []
    blink_count = 0
    in_blink = False
    blink_start = None

    for i in range(num_frames):
        values = selectIndices(raw_dict, i)
        x1, y1 = values[0]  
        x2, y2 = values[1] 
        distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

        if distance < average_distance * 0.45:
            if not in_blink:
                in_blink = True  # Blink start
                blink_start = i
        else:
            if in_blink:
                # Blink end
                in_blink = False
                blink_indices.append((blink_start, i - 1))

    # Check if the last frame is part of an ongoing blink
    if in_blink:
        blink_indices.append((blink_start, num_frames - 1))

    blink_count = len(blink_indices)
    return blink_indices, blink_count

"""
Draw MatPlotLib animation.
:param graph_eye_choice: String 'left', 'right', or 'both' eyes.
:param interval: duration of each frame in milliseconds
:param slow_down_multiplier: how many times to repeat blink frames (slows down animation when blink event is met)
"""
def generate_eye_animation(selectIndices, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=50, slow_down_multiplier=4):
    
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
        distanceLine, = ax.plot([], [], 'bo-')

    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    blink_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    blink_true = ax.text(0.1, 0.9, '', transform=ax.transAxes, color = "green")
    blink_false = ax.text(0.1, 0.9, '', transform=ax.transAxes, color = "red")
    blink_count_text = ax.text(0.2, 0.95, '', transform=ax.transAxes)
    current_area_text = ax.text(0.2, 0.9, '', transform=ax.transAxes)

    # Function to initialize plot
    def init():
        frame_text.set_text('')
        blink_text.set_text('')
        if graph_eye_choice == "both":
            line_left.set_data([], [])
            line_right.set_data([], [])
            return line_left, line_right, frame_text
        else:
            line.set_data([], [])
            distanceLine.set_data([], [])
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
        values = selectIndices(raw_dict,frame)
        x1, y1 = values[0]  
        x2, y2 = values[1] 
        current_distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

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
            current_area_text.set_text(f'Area: {current_distance * 1000:.5f}')
            return line_left, line_right, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text
        elif graph_eye_choice == "left":
            vertices = frames_list_left[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_distance * 1000:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text
        elif graph_eye_choice == "right":
            vertices = frames_list_right[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_distance * 1000:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text

    # Calculate total number of frames including repeated frames for slow down
    extra_frames_for_blinks = sum((end - start + 1) * (slow_down_multiplier - 1) for start, end in blink_indices)
    num_frames = len(frames_list_left) + extra_frames_for_blinks

    animation = FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=True, interval=interval, repeat=False)

    plt.show()

if __name__ == "__main__":
    main()