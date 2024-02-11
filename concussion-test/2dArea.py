
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

    start_time = time.time()
    blink_indices, blink_count = detect_blinks(raw_dict, num_frames, selectIndices)
    end_time = time.time()

    print(blink_indices)
    print(blink_count)
    print("Elapsed Time:", (end_time - start_time) * 1000, "ms")

    generate_eye_animation(selectIndices, polyArea, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=150, slow_down_multiplier=1)

def selectIndices(raw_dict, frame, close_loop = False, eye_choice = "left"):
    """
    Construct a new list of X and Y values of each eyelid landmark for a given frame
 
    :param frame: The indexed frame
    :param raw_dict: The current dictionary being read
    :param close_loop: Boolean flag indicating whether to append first index to end for animation
    :param eye_choice: String 'left' or 'right' eye
    :return: array of vertices for each eyelid landmark
    """ 

    # indexes for other eye: indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7] [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]

    if eye_choice == "left":
        eye_indices = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    elif eye_choice == "right":
        eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    else:
        print("Invalid eye choice")
        return

    selected_vertices = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]

    if close_loop and eye_choice == "left":
        selected_vertices_left = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_left.append(raw_dict['apps']['frames'][frame]['mp'][263][:2])
        return np.array(selected_vertices_left)
    elif close_loop and eye_choice == "right":
        selected_vertices_right = [raw_dict['apps']['frames'][frame]['mp'][index][:2] for index in eye_indices]
        selected_vertices_right.append(raw_dict['apps']['frames'][frame]['mp'][33][:2])
        return np.array(selected_vertices_right)

    return np.array(selected_vertices)

def polyArea(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                        for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def detect_blinks(raw_dict, num_frames, selectIndices):

    #get amount of frames
    num_frames = len(raw_dict['apps']['frames'])

    #calculate average area
    sum_area = 0
    for i in range(num_frames):
        list = selectIndices(raw_dict, i, eye_choice = "left")
        sum_area += polyArea(list)
    average_area = sum_area / num_frames
    print(f'Average Area: {average_area:.5f}')

    #compare current area to average area
    sum_area = 0
    blink_count = 0
    blink_indices = []
    in_blink = False
    for i in range(num_frames):
        list_left = selectIndices(raw_dict, i, eye_choice = "left")
        current_area = (polyArea(list_left))

        if current_area < average_area - 0.008:
            if not in_blink:
                in_blink = True
                blink_start = i
            # If it's the last frame, close the ongoing blink
            if i == num_frames - 1:
                blink_indices.append((blink_start, i))
        else:
            if in_blink:
                in_blink = False
                blink_indices.append((blink_start, i - 1))
    
    blink_count = len(blink_indices)
    return blink_indices, blink_count

def generate_eye_animation(selectIndices, polyArea, raw_dict, num_frames, blink_indices, graph_eye_choice="both", interval=50, slow_down_multiplier=4):
    
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
        
        # Calculate current area to display on graph
        current_vertices = selectIndices(raw_dict, current_frame[0], eye_choice="left")
        current_area = polyArea(current_vertices)

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
            current_area_text.set_text(f'Area: {current_area:.5f}')
            return line_left, line_right, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text
        elif graph_eye_choice == "left":
            vertices = frames_list_left[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_area:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text
        elif graph_eye_choice == "right":
            vertices = frames_list_right[frame]
            x, y = zip(*vertices)
            line.set_data(x, y)
            blink_text.set_text("Blink: ")
            frame_text.set_text(f'Frame: {frame}')
            completed_blinks = sum(end < frame for start, end in blink_indices)
            blink_count_text.set_text(f'Blink Count: {completed_blinks}')
            current_area_text.set_text(f'Area: {current_area:.5f}')
            return line, frame_text, blink_text, blink_true, blink_false, blink_count_text, current_area_text

    # Calculate total number of frames including repeated frames for slow down
    extra_frames_for_blinks = sum((end - start + 1) * (slow_down_multiplier - 1) for start, end in blink_indices)
    num_frames = len(frames_list_left) + extra_frames_for_blinks

    animation = FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=True, interval=interval, repeat=False)

    plt.show()

if __name__ == "__main__":
    main()