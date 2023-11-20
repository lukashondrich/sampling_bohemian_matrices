import cv2
import numpy as np
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import cProfile
from joblib import Parallel, delayed, cpu_count



def sample_from_unit_circle():
    """
    Sample a complex number from the unit circle.
    """
    angle = np.random.uniform(0, 2*np.pi)
    return np.exp(1j * angle)

def interpolate_matrices(matrix_0, matrix_1, alpha):
    interpolated_matrix = (1 - alpha) * matrix_0 + alpha * matrix_1
    return interpolated_matrix

def sample_from_interpolated_matrix(interpolated_matrix):
    t1 = sample_from_unit_circle()
    t2 = sample_from_unit_circle()
    interpolated_matrix[0, 0] = t1
    interpolated_matrix[3, 3] = t2
    return interpolated_matrix

def compute_eigenvalues(interpolated_matrix):
    interpolated_matrix = sample_from_interpolated_matrix(interpolated_matrix)
    eigenvalues = np.linalg.eigvals(interpolated_matrix)
    return eigenvalues

def plot_eigenvalues(matrix, eigenvalues_list, fig, ax, frames,i, desired_shape=(7680, 4320)):
    ax.clear()
    ax.axis('off')  
    # for eigenvalues in eigenvalues_list:
    all_eigenvalues = np.array(eigenvalues_list)
    ax.scatter(all_eigenvalues.real, all_eigenvalues.imag, marker='.', s=0.09, color='red')
    ax.set_xlim([-3, 3])  
    ax.set_ylim([-3, 3])  
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.grid(False)  
    
    

    
    # Create an inset to display the matrix
    inset_ax = fig.add_axes([0.75, 0.75, 0.2, 0.2])  # Adjust the position and size as needed
    inset_ax.matshow(np.abs(matrix), cmap='viridis')  # Use the absolute values of the matrix
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    
    
    
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    
    
    # Determine the shape based on the size of the image data
    height = int(image.shape[0] / (3 * desired_shape[0]))
    image = image.reshape((height, desired_shape[0], 3))
    
    # Resize the image to the desired shape
    image = cv2.resize(image, desired_shape)
    cv2.imwrite(f'/Users/lukashondrich/Documents/bohemian_matrix_video/frame_{i}.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    frames.append(image)
    
    return frames
    
def make_frames(size, matrix_0, matrix_1, num_interpolations, num_samples_per_interpolation):
    frames = []
    desired_shape = size # Desired width and height 4k resolution: 3840 x 2160, 8k: 7680 x 4320
    
    
    fig, ax = plt.subplots(figsize= (desired_shape[0] / 100, desired_shape[1] / 100), dpi=50)

    for i in range(num_interpolations): 
        print(f"Generating frame {i+1}/{num_interpolations}")
        import time
        if i>0: 
            time_elapsed = time.time() - time_start
        time_start = time.time()
                
        alpha = i / num_interpolations
    
        interpolated_matrix = interpolate_matrices(matrix_0, matrix_1, alpha)
        eigenvalues_list = Parallel(n_jobs=-1)(delayed(compute_eigenvalues)(interpolated_matrix) for _ in range(num_samples_per_interpolation)) 
        #print number of cores used:
        print(f"Number of cores used: {cpu_count()}")
        
        plot_eigenvalues(interpolated_matrix, eigenvalues_list, fig, ax, frames, i, desired_shape=desired_shape)

    print(f"Number of frames: {len(frames)}")
    return frames
        

def make_video(pathOut,fps, size, matrix_array, num_interpolations, num_samples_per_interpolation):
    """_summary_

    """
    # use the matrices as waypoints for the interpolation back to the first matrix
    
    #0 to 1
    frame_array = make_frames(size, matrix_array[0], matrix_array[1], num_interpolations, num_samples_per_interpolation)
    
    if len(matrix_array) > 2: 
        #loop through the rest and start with i = 1
        for i in range(1, len(matrix_array)-1):
            frame_array.extend(make_frames(size, matrix_array[i], matrix_array[i+1], num_interpolations, num_samples_per_interpolation))

    frame_array.extend(make_frames(size, matrix_array[-1], matrix_array[0], num_interpolations, num_samples_per_interpolation))
        
    
    print(f"Number of frames: {len(frame_array)}") 
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    #out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'H264'), fps, size) 

        
    for i in range(len(frame_array)):
        #load frame 
        #filename = f'/Users/lukashondrich/Documents/bohemian_matrix_video/frame_{i}.png'
        #img = cv2.imread(filename)
        
        ## append from frame_array
        img = frame_array[i]
        
        # writing to image array
        out.write(img)
        print(f"Writing frame {i+1}/{len(frame_array)}")
    out.release()





def main():
    pathOut = '/Users/lukashondrich/Documents/bohemian_matrix_video/bohemian.mp4'
    fps = 30.0
    size = (7680, 4320) #(3840, 2160) # (1024, 730)
    matrix_0 = np.array([
    [1, -1, -1j, -1j, -1j],
    [-1, 1, 1j, -1, 1j],
    [1, 1j, 0, 1, 0],
    [-1, -1j, -1, 2, -1],
    [0, 0, 1, 0, 1]
    ])

    matrix_1 = np.array([
        [-1, 1, 1, 1j, -1],
        [1, 1, 0, -1, -1j],
        [-1, 1j, 3, -1, 1j],
        [1j, -1, -1, 0, -1],
        [1, 1, 4, 1, -1]
    ])    
    matrix_1 = np.array([
        [0, 1, -1, 8, 0],
        [0, 1, 0, 1, 0],
        [-1, 1, -1, 1, -1],
        [0, 1, 0, 1, 0],
        [0, 1j, -1, 1j, 0]
    ])
    matrix_2= np.array([
        [0, 1, -1, 8, 4],
        [0, 1, 0, 1, 4],
        [-1, 1, -1, 1, -1],
        [0, 1, 0, 1, 4],
        [0, 1j, -1, 1j, 4]
    ])
    
    matrix_array = [matrix_0, matrix_1, matrix_2]

    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    make_video(pathOut,fps, size, matrix_array, 30, 3_000)
    profiler.disable()
    profiler.dump_stats('profile_results_parallel.prof')
    #profiler.print_stats(sort='cumulative')
    
    # save profiler output: 



    
if __name__== "__main__":
    main()
    
    
    
