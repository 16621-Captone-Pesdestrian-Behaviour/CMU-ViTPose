import scipy.io as sio
import numpy as np
import cv2
import os

COLOR_LIST = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
              (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), 
              (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
              (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,0,0)
thickness              = 2
lineType               = 2

def load_mat_dict(mat_path):
    return sio.loadmat(mat_path)

def video_to_frame(video_path, output_path, start_frame=0):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count >= start_frame:
            write_path = os.path.join(output_path, f"frame_{str(count).zfill(4)}.png" )
            cv2.imwrite(write_path, image)     
            success,image = vidcap.read()
            if count % 600 == 0:
                print(f"finished frame {count}")
        count += 1

def make_videos(frames, output_path, fps=60):
    """Write a video given list of frames.

    Args:
        frames: list
        output_path: path to write the video
        fps: fps to write the video
    """
    out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, frames[0].shape[:2][::-1])
    for f in frames:
        out.write(f)
    out.release()

def get_param( view_num, video_num, tracker_id ):
    """Get intrinsic matrix, homography, tracklet's starting frame, video's starting frame, tracklet's trajectories.
    
    Args:
        view_num: 1, 2, or 3 which camera's view.
        video_num: 0-7, video clip.
        tracker_id: tracklet's label. 
    
    Returns:
        K: Intrinsic matrix for the camera view_num.
        H: Homography to map from z=0 plane to camera's image plane.
        tracklet_start_frame: First frame to start tracking the tracklet after synchonize.
        start_frame: The fisrt frame for the video, used to synchronize the frames.
        trajectory: 3D trajectory on z=0 plane.
    """
    if view_num == 0:
        raise ValueError("view num is 1-index, got 0")

    K = load_mat_dict(f"./video_data_n{view_num}/C0.mat")["K"]
    k = load_mat_dict(f"./video_data_n{view_num}/C0.mat")["k"]
    H = load_mat_dict(f"./video_data_n{view_num}/H0.mat")["H"]
    start_frame_txt_path = f"./video_data_n{view_num}/start_frames.txt"
    with open(start_frame_txt_path) as f:
        start_frame_lines = [line for line in f.readlines()]
    start_frame = int(start_frame_lines[video_num][2:])

    traj_start_frames = load_mat_dict("./label/0.mat")
    
    from pdb import set_trace; set_trace()
    
    traj_start_frames = [s[0][0][0] for s in traj_start_frames["traj_starts"].T]

    trajectory = load_mat_dict(f"./label/{video_num}.mat")["trajectories"][0]
    if type(tracker_id) is list:
        traj_start_frames = [ traj_start_frames[_id] for _id in tracker_id ]
        trajectory = trajectory[tracker_id]
    elif tracker_id >= 0:
        traj_start_frames = traj_start_frames[tracker_id]
        trajectory = trajectory[tracker_id]

    return K, k, H, traj_start_frames, start_frame, trajectory

def project_trajectories(trajectories, H):
    """Project trajectories on z=0 plane to camera frame given H.

    Args:
        trajectories: Nx3, 3d coordinate [x, y, 0].
        H: Homography.

    Returns:
        trajectories: Nx2, 2d [x, y] coordinate on new plane.
    """
    trajectories = np.hstack( [trajectories[:, :2], np.ones( (trajectories.shape[0], 1) )])
    trajectories = H.dot( trajectories.T ).T
    trajectories = trajectories/trajectories[:, -1, None]
    return trajectories[:, :2].astype(int)

def read_frame(video_dir, frame_path):
    """Tracking trajectory can be obtained with 2 camera view, but we have 3 cameras. So for some camera, it is possible to have trajectory
    without a valid frame. This function checks if the frame exist, is not, read the last exist frame.

    Args:
        video_dir: Directory of frames.
        frame_path: Path to read.

    Returns:
        np.array for read frame.    
    """
    if not os.path.exists(frame_path):
        frame_path = os.path.join( video_dir, sorted(os.listdir(video_dir))[-1] )
    return cv2.imread(frame_path)

def draw_trajtory( video_dir, output_dir, view_num, video_num, tracker_id):
    video_dir =  video_dir.format( view_num, video_num )
    output_dir = output_dir.format( view_num )

    print(f"drawing {view_num}, video_num {video_num}, label {tracker_id}")
    if type(tracker_id) is list or tracker_id == -1:
        multi_track = True
    else:
        multi_track = False

    K, k, H, tracklet_start_frame, video_start_frame, trajectories = get_param(view_num = view_num, 
                                                                            video_num = video_num, 
                                                                            tracker_id = tracker_id)
    # print(K, H, tracklet_start_frame, video_start_frame)
    
    if multi_track:
        num_of_frames = [len(_traj) for _traj in trajectories]
        last_frame_num = max( [_label_s + _nof + video_start_frame - 1 for _label_s, _nof in zip(tracklet_start_frame, num_of_frames)] ) 
        trajectories = [project_trajectories(_traj, H) for _traj in trajectories] 
    else:
        num_of_frames = len(trajectories)
        last_frame_num = tracklet_start_frame + num_of_frames + video_start_frame - 1
        trajectories = project_trajectories(trajectories, H)

    # due to sync, some view might not contain frame corresponds to last few trajectries.
    last_frame_path = os.path.join(video_dir, f"frame_{str(last_frame_num).zfill(4)}.png")
    last_frame = read_frame(video_dir, last_frame_path)

    if write_intermediate:
        all_frames = []
        if multi_track:
            first_frame_num = min(tracklet_start_frame)
            # tracklet_end_frame -> include
            tracklet_end_frame = [ _start + _nof - 1 for _start, _nof in zip(tracklet_start_frame, num_of_frames) ]
            last_frame_num = max(tracklet_end_frame)
            # Walk through frames
            # frame num in tracklet space
            for i in range( first_frame_num, last_frame_num+1 ):

                current_frame_num = i + video_start_frame
                current_frame_path = os.path.join( video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png"  )
                current_frame = read_frame(video_dir, current_frame_path) # use read frame to prevent exceed time frame

                if (i-first_frame_num)%100 == 0:
                    print(f"Current process: frame num: {current_frame_num}, {i-first_frame_num}/{last_frame_num-first_frame_num}")

                # Walk through each tracklet
                for _label_id, (_start, _end) in enumerate(zip(tracklet_start_frame, tracklet_end_frame)):
                    if  _start <= i and _end >= i:
                        image_coor = trajectories[_label_id][current_frame_num - _start - video_start_frame ]
                        cv2.circle(current_frame, image_coor, 5, COLOR_LIST[_label_id%len(COLOR_LIST)], -1)

                imwrite_path = os.path.join(output_dir, f"frame{current_frame_num}.png")
                all_frames.append(current_frame)
                cv2.putText(current_frame, f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                cv2.imwrite(imwrite_path, current_frame)

        else:
            for i, traj in enumerate(trajectories):
                image_coor = traj
                current_frame_num = i + tracklet_start_frame + video_start_frame
                current_frame_path = os.path.join( video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png"  )
                current_frame = read_frame(video_dir, current_frame_path)

                cv2.circle(current_frame, image_coor, 2, (0, 255, 0), -1)
                all_frames.append(current_frame)
                imwrite_path = os.path.join(output_dir, f"frame{current_frame_num}.png")
                cv2.putText(current_frame, f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                cv2.imwrite(imwrite_path, current_frame)

        make_videos( all_frames, f"view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi", fps = 60 )
        print(f"Write to video path: view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi")

    for i, _traj in enumerate(trajectories):
        if multi_track:
            for _coor in _traj:
                cv2.circle(last_frame, _coor,  2, COLOR_LIST[i%len(COLOR_LIST)], -1)
        else:
            cv2.circle(last_frame, _traj,  2, (0, 255, 0), -1)

    cv2.imwrite(f"./whole_traj/last_frame_{view_num}_video_{video_num}_label_{tracker_id}.png", last_frame)

def triangulate( pt1, pt2, k1, k2, ex1, ex2 ):
    """Get 3D coordinate through triangulation.

    Args:
        x1: point on perspective 1
        x2: point on perspective 2
        k1: intrinsic for camera 1
        k2: intrinsic for camera 2
        ex1: extrinsic for camera 1
        ex2: extrinsic for camera 2
    
    return:
        X: point in 3D    
    """

    x1, y1 = pt1
    x2, y2 = pt2
    p1 = k1@ex1
    p2 = k2@ex2
    A = np.array( [ [y1*p1[1] - p1[2]],
                    [p1[0] - x1*p1[2]],
                    [y2*p2[1] - p2[2]],
                    [p1[0] - x2*p2[2]]] )
    _, _, VT = np.linalg.svd( A )
    X = VT[-1]
    return X/X[-1]

if __name__ == "__main__":
    video_to_frame("./video_data_n1/0.mp4", "./video_data_n1/tepper_0_frames", start_frame=1294)
    video_to_frame("./video_data_n2/0.mp4", "./video_data_n2/tepper_0_frames", start_frame=1316)
    video_to_frame("./video_data_n3/0.mp4", "./video_data_n3/tepper_0_frames", start_frame=1320)

    # view_num = 2
    # video_num = 0
    # tracker_id = 4
    # write_intermediate = True
    # for view in range(1, 4):
    #     draw_trajtory( view, video_num, tracker_id)
    # write_intermediate = True
    # video_dir =  "./video_data_n{}/tepper_{}_frames"
    # output_dir = "./test_output_n{}"
    # draw_trajtory( video_dir, output_dir, view_num, video_num, [0, 1, 2, 3])
    # draw_trajtory( video_dir, output_dir, view_num, video_num, 0)
    # draw_trajtory( video_dir, output_dir, view_num, video_num, -1)
    # for i in range(4):
    #     draw_trajtory(view_num, video_num, i)
    # for view in range(1, 4):
    #     for tracker_id in range(14, 18):
    #         draw_trajtory( view, video_num, tracker_id)

    # extrinsic = load_mat_dict("./extrinsics_sess_0_1_2_3.mat")
    # view 1, 2, 3 -> left, right, center
    # R1, R2, R3 = extrinsic["R_left"], extrinsic["R_right"], extrinsic["R_center"]
    # t1, t2, t3 = extrinsic["t_left"], extrinsic["t_right"], extrinsic["t_center"]
    # ex1 = np.hstack( [ R1, t1 ] )
