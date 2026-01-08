# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# new imports (not from UMI code)
from ast import literal_eval as make_tuple

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

# constants
tac02_res = (66,)

# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-org', '--out_res_gelsight', type=str, default='224,224')
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers, out_res_gelsight):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))
    out_res_gelsight = tuple(int(x) for x in out_res_gelsight.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        
        videos_dict = defaultdict(list)
        for plan_episode in plan:
            grippers = plan_episode['grippers']
            
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
            
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                gelsight_path = camera['gelsight_path']
                tac02_path = camera['tac02_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                assert gelsight_path.is_file()
                assert tac02_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'gelsight_path': gelsight_path,
                    'tac02_path': tac02_path,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start,
                    'video_timestamps': plan_episode['episode_timestamps'],
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
        
    # get gelsight image size
    with av.open(str(vid_args[0][1][0]['gelsight_path'])) as container:
        in_stream = container.streams.video[0]
        gelsight_ih, gelsight_iw = in_stream.height, in_stream.width
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        _ = out_replay_buffer.data.require_dataset(
            name=f'camera{cam_id}_rgb',
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
        _ = out_replay_buffer.data.require_dataset(
            name=f'gelsight{cam_id}_rgb',
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res_gelsight + (3,),
            chunks=(1,) + out_res_gelsight + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
        _ = out_replay_buffer.data.require_dataset(
            name=f'tac02_{cam_id}',
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + tac02_res,
            chunks=(1,) + tac02_res,
            dtype=np.uint16,
        )


    def video_to_zarr(replay_buffer, mp4_path, tasks):
        assert len(tasks) == 1, "TMI only supports single device training for now"
        gelsight_path = tasks[0]['gelsight_path']
        tac02_path = tasks[0]['tac02_path']
    
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        resize_tf_gelsight = get_image_transform(
            in_res=(gelsight_iw, gelsight_ih),
            out_res=(out_res_gelsight[1], out_res_gelsight[0])
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        gelsight_name = f'gelsight{camera_idx}_rgb'
        tac02_name = f'tac02_{camera_idx}'
        gelsight_img_array = replay_buffer.data[gelsight_name]
        tac02_array = replay_buffer.data[tac02_name]

        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        gelsight_container = av.open(str(gelsight_path))
        gelsight_in_stream = gelsight_container.streams.video[0]
        
        tac02_datafile = open(tac02_path, "r")
        tac02_data = tac02_datafile.read()
        
        tac02_time_and_frames = tac02_data.split("\n")[:-1]
        tactile_timestamps = []
        tac02_taxel_data = []
        video_timestamps = tasks[0]['video_timestamps']
        for data in tac02_time_and_frames:
            timestamp, tac02_taxels = data.split(";")
            timestamp = float(timestamp) / 1e9
            
            # Thanks to https://stackoverflow.com/questions/9763116/parse-a-tuple-from-a-string
            # I found a way to parse a tuple without eval()
            tac02_taxels = make_tuple(tac02_taxels)
            tac02_taxels = np.array(tac02_taxels).reshape(tac02_res)
            
            # If I want to normalize the TAC-02 data, it should be done here.
            tac02_taxels = 3500 - tac02_taxels
            
            tactile_timestamps.append(timestamp)
            tac02_taxel_data.append(tac02_taxels)
        
        with av.open(mp4_path) as container:
            gelsight_video_frames = gelsight_container.decode(gelsight_in_stream)
            cur_tactile_idx = 0
            last_opened_gelsight_idx = -1
        
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            gelsight_frames = enumerate(gelsight_container.decode(gelsight_in_stream))
            cur_gelsight_frame_idx_check, cur_gelsight_frame = next(gelsight_frames)
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                    
                gelsight_path = tasks[curr_task_idx]["gelsight_path"]
                tac02_path = tasks[curr_task_idx]["tac02_path"]
                start_idx = tasks[curr_task_idx]['frame_start']
                end_idx = tasks[curr_task_idx]['frame_end']
                
                # To ensure both tactile and video input start at about the same time
                assert abs(tactile_timestamps[0] - video_timestamps[0]) <= 5
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    while cur_tactile_idx+1 < gelsight_in_stream.frames and (tactile_timestamps[cur_tactile_idx+1] < video_timestamps[frame_idx-start_idx]):
                        cur_tactile_idx += 1
                        cur_gelsight_frame_idx_check, cur_gelsight_frame = next(gelsight_frames)
                        assert cur_tactile_idx == cur_gelsight_frame_idx_check
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')
                    gelsight_img = cur_gelsight_frame.to_ndarray(format='rgb24')
                    tac02_taxels = tac02_taxel_data[cur_tactile_idx]

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)

                    # resize gelsight image
                    gelsight_img = resize_tf_gelsight(gelsight_img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]

                    # compress image
                    img_array[buffer_idx] = img
                    gelsight_img_array[buffer_idx] = gelsight_img
                    tac02_array[buffer_idx] = tac02_taxels
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
