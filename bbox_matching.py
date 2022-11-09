import scipy.io as sio
import numpy as np
import cv2
import os
from time import time
from run import DetModel, PoseModel
from tqdm import tqdm
import pickle
import argparse

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

det_model = DetModel("cuda")
pose_model = PoseModel("cuda")

def _match_bbox(img, point, viz=False):
    bboxs = det_model.detect(image=img, score_thr=0.3)
    pose_out = pose_model.predict_pose(img, bboxs)
    bbox = np.array([pose["bbox"] for pose in pose_out])
    a = bbox[:, [0, 2]].mean(axis=1)
    b = bbox[:, [1, 3]].mean(axis=1)
    c = np.vstack((a, b)).T
    min_dist_idx = np.argmin(np.hypot(*(c - point).T))
    res = [pose_out[min_dist_idx]]
    if viz:
        # img = cv2.circle(img, p, 1, (255, 0, 0), 2)
        img = pose_model.visualize_pose_results(image=img, pose_results=res)
        # cv2.imwrite("out_img.png", img)
    return img, res

def IoU(bboxes1, bboxes2):
    bboxes1 = bboxes1[:, :4]
    bboxes2 = bboxes2.reshape((1, 5)) 
    bboxes2 = bboxes2[:, :4]
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou

def in_box(box, point):
    x1, y1, x2, y2 = box[:4]
    xp1, yp1 = point[:2]
    if x1<=xp1<=x2 and y1<=yp1<yp2:
        return True

def match_bbox(img, point, pose_out, prev_box=None, viz=False):
    bbox = np.array([pose["bbox"] for pose in pose_out])
    if prev_box is None:
        a = bbox[:, [0, 2]].mean(axis=1)
        b = bbox[:, [1, 3]].mean(axis=1)
        c = np.vstack((a, b)).T
        min_dist_idx = np.argmin(np.hypot(*(c - point).T))
        res = [pose_out[min_dist_idx]]
    else:
        ious = IoU(bboxes1=bbox, bboxes2=prev_box)
        max_iou_idx = np.argmax(ious)
        res = [pose_out[max_iou_idx]]
    if viz:
        # img = cv2.circle(img, p, 1, (255, 0, 0), 2)
        img = pose_model.visualize_pose_results(image=img, pose_results=res)
        # cv2.imwrite("out_img.png", img)
    prev_box = res[0]["bbox"]
    return img, prev_box, res

def video_to_frame(video_path, output_path, start_frame=0):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count >= start_frame:
            write_path = os.path.join(output_path, f"frame_{str(count).zfill(4)}.png" )
            cv2.imwrite(write_path, image)     
            success,image = vidcap.read()
            if count % 500 == 0:
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

    K = load_mat_dict(f"./RGB_videos/video_data_n{view_num}/C{video_num}.mat")["K"]
    k = load_mat_dict(f"./RGB_videos/video_data_n{view_num}/C{video_num}.mat")["k"]
    H = load_mat_dict(f"./RGB_videos/video_data_n{view_num}/H{video_num}.mat")["H"]
    start_frame_txt_path = f"./RGB_videos/video_data_n{view_num}/start_frames.txt"
    with open(start_frame_txt_path) as f:
        start_frame_lines = [line for line in f.readlines()]
    start_frame = int(start_frame_lines[video_num][2:])

    traj_start_frames = load_mat_dict(f"./labels/{video_num}.mat")
    
    traj_start_frames = [s[0][0][0] for s in traj_start_frames["traj_starts"].T]

    trajectory = load_mat_dict(f"./labels/{video_num}.mat")["trajectories"][0]
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



def draw_trajtory(video_dir, output_dir, view_num, video_num, tracker_id, save_dict, write_intermediate):
    video_dir =  video_dir.format(view_num, video_num)
    output_dir = output_dir.format(view_num)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    print(f"drawing {view_num}, video_num {video_num}, label {tracker_id}")
    if type(tracker_id) is list or tracker_id == -1:
        multi_track = True
    else:
        save_dict[video_num][view_num][tracker_id] = save_dict[video_num][view_num].get(tracker_id, [])
        multi_track = False

    K, k, H, tracklet_start_frame, video_start_frame, trajectories = get_param(view_num = view_num, 
                                                                            video_num = video_num, 
                                                                            tracker_id = tracker_id)
    
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
        prev_bbox = {}
        all_frames = []
        if multi_track:
            first_frame_num = min(tracklet_start_frame)
            # tracklet_end_frame -> include
            tracklet_end_frame = [ _start + _nof - 1 for _start, _nof in zip(tracklet_start_frame, num_of_frames) ]
            last_frame_num = max(tracklet_end_frame)
            # Walk through frames
            # frame num in tracklet space
            for i in tqdm(range(first_frame_num, last_frame_num+1)):

                current_frame_num = i + video_start_frame
                current_frame_path = os.path.join( video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png")

                current_frame = read_frame(video_dir, current_frame_path)
                cv2.putText(current_frame.astype(np.uint8), f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                bboxs = det_model.detect(image=current_frame, score_thr=0.4)
                pose_out = pose_model.predict_pose(current_frame, bboxs)
                current_bbox = np.array([pose["bbox"] for pose in pose_out])

                if (i-first_frame_num)%100 == 0:
                    print(f"Current process: frame num: {current_frame_num}, {i-first_frame_num}/{last_frame_num-first_frame_num}")

                # Walk through each tracklet
                for _label_id, (_start, _end) in enumerate(zip(tracklet_start_frame, tracklet_end_frame)):
                    if  _start <= i and _end >= i:
                        save_dict[video_num][view_num][_label_id] = save_dict[video_num][view_num].get(_label_id, [])
                        image_coor = trajectories[_label_id][current_frame_num - _start - video_start_frame ]
                        cv2.circle(current_frame.astype(np.uint8), image_coor, 5, COLOR_LIST[_label_id%len(COLOR_LIST)], -1)
                        prev_bbox[_label_id] = prev_bbox.get(_label_id, None)
                        current_frame, prev_box, res = match_bbox(current_frame, image_coor, pose_out, prev_bbox[_label_id], True)
                        save_dict[video_num][view_num][_label_id].append(res[0])
                        prev_bbox[_label_id] = prev_box

                imwrite_path = os.path.join(output_dir, f"frame_{str(current_frame_num).zfill(4)}.png")
                all_frames.append(current_frame) 
                cv2.imwrite(imwrite_path, current_frame)

        else:
            for i, traj in enumerate(tqdm(trajectories)):
                image_coor = traj
                current_frame_num = i + tracklet_start_frame + video_start_frame
                current_frame_path = os.path.join(video_dir,  f"frame_{str(current_frame_num).zfill(4)}.png")

                current_frame = read_frame(video_dir, current_frame_path)
                bboxs = det_model.detect(image=current_frame, score_thr=0.4)
                pose_out = pose_model.predict_pose(current_frame, bboxs)
                current_bbox = np.array([pose["bbox"] for pose in pose_out])
                prev_bbox[tracker_id] = prev_bbox.get(tracker_id, None)

                cv2.circle(current_frame.astype(np.uint8), image_coor, 2, (0, 255, 0), -1)
                cv2.putText(current_frame.astype(np.uint8), f"{video_start_frame}/{current_frame_num}", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                current_frame, prev_box, res = match_bbox(current_frame, image_coor, pose_out, prev_bbox[tracker_id], True)
                save_dict[video_num][view_num][tracker_id].append(res[0])
                prev_bbox[tracker_id] = prev_box

                all_frames.append(current_frame)
                imwrite_path = os.path.join(output_dir, f"frame_{str(current_frame_num).zfill(4)}.png")
                cv2.imwrite(imwrite_path, current_frame)
        #     print(f"Length Traj: {len( save_dict[video_num][view_num][tracker_id])}")
        # print(f"Length: {len(all_frames)}")
        make_videos( all_frames, f"view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi", fps = 60 )
        print(f"Write to video path: view_{view_num}_video_{video_num}_track_id_{tracker_id}.avi")

    for i, _traj in enumerate(trajectories):
        if multi_track:
            for _coor in _traj:
                cv2.circle(last_frame, _coor,  2, COLOR_LIST[i%len(COLOR_LIST)], -1)
        else:
            cv2.circle(last_frame, _traj,  2, (0, 255, 0), -1)

    cv2.imwrite(f"./whole_traj/last_frame_{view_num}_video_{video_num}_label_{tracker_id}.png", last_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_num", type=int, default=0)
    parser.add_argument("--view_num", type=int, default=1)
    parser.add_argument("--tracker_id", type=int, default=-1)
    # parser.add_argument("--tracker_id", type=list, default=[])
    args = parser.parse_args()

    video_dir =  "./RGB_videos/video_data_n{}/tepper_{}_frames"
    output_dir = "./RGB_videos/test_output_n{}"
    write_intermediate = True

    video_num = args.video_num
    view_num = args.view_num
    tracker_id = args.tracker_id
    save_dict = {}
    save_dict[video_num] = {1: {}, 2: {}, 3: {}}

    draw_trajtory(video_dir, output_dir, view_num, video_num, tracker_id, save_dict, write_intermediate)

    # f = open(f'{video_num}_{view_num}.pickle', 'wb')
    # pickle.dump(save_dict, f)
    # f.close()
