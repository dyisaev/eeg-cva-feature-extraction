import json
from pathlib import Path
import subprocess
import os
import pandas as pd

OPENFACE_EXEC = '/media/st4Tb/projects/OpenFace/build/bin/FaceLandmarkImg'
def get_video_fps(filename):
    result=subprocess.run(["ffprobe","-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1",
                           "-show_entries", "stream=r_frame_rate", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return eval(result.stdout)

def create_random_folder(temproot):
    folder=str.strip(subprocess.run(['mktemp','-d',f'--tmpdir={temproot}'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT).stdout.decode('utf-8'))
    subprocess.run(['chmod', '766',folder])
    return folder
def main(bboxes_file,input_video_file,output_video_folder,output_csv_folder,output_json_folder,temp_root_folder):
    
    bboxes=json.load(open(bboxes_file,'r'))
    '''
    videoframes_folder=create_random_folder(temp_root_folder)
    # extract frames from video
    subprocess.run(['ffmpeg','-i',input_video_file,videoframes_folder+'/%010d.jpg'])
    
    # ffmpeg creates frames starting from 1, while in bboxes frames start from 0

    # create frames bboxes from json
    frames_folder=create_random_folder(temp_root_folder)

    for frame,left,right,top,bottom in zip(bboxes['frame'],bboxes['left'],bboxes['right'],bboxes['top'],bboxes['bottom']):
      with open(frames_folder+'/'+f'{(frame+1):010d}.txt','w') as f:
        print(f'{left} {top} {right} {bottom}', file=f)

    noface_frames_folder=create_random_folder(temp_root_folder)
        
    img_files = os.listdir(videoframes_folder)
    img_files = [videoframes_folder+'/'+f for f in img_files if os.path.isfile(videoframes_folder+'/'+f)]

    # temporarily remove the images that do not have corresponding bboxes (face is not detected) - this is neeeded for OpenFace
    for img_file in img_files:
        fname=Path(img_file).stem
        frame_file=frames_folder+'/'+fname+'.txt'
        if Path(frame_file).is_file():
            continue
        else:
            noface_img_file=noface_frames_folder+'/'+fname+'.jpg'
            os.replace(img_file,noface_img_file)
    
    tmp_openface_output_folder=create_random_folder(temp_root_folder)

    print('videoframes_folder: ', videoframes_folder)
    print('bbox_folder: ', frames_folder)
    print('noface_frames_folder: ', noface_frames_folder)
    print('openface_output_folder: ', tmp_openface_output_folder)

    subprocess.run([OPENFACE_EXEC, '-wild', '-fdir', videoframes_folder,'-bboxdir', frames_folder,
                     '-out_dir' ,tmp_openface_output_folder])
    

    #copy back to openface output frames where face is not detected
    subprocess.run(['cp', noface_frames_folder+'/*.jpg', tmp_openface_output_folder+'/'])
    
    fps = str(get_video_fps(input_video_file))
    print('original video fps: ', fps)
   
    # get together a video with landmarks:
    subprocess.run(['ffmpeg' ,'-framerate', fps,  '-i' ,tmp_openface_output_folder+'/%010d.jpg',  
                    '-c:v' ,'libx264'  ,'-pix_fmt', 'yuv420p', '-r', fps ,output_video_folder+'/landmarks_video.mp4'])

    # get together features:
    '''
    tmp_openface_output_folder='/home/disaev/tmp/tmp.7aceOIDnyf'
    csv_arr=[]
    for frame in bboxes['frame']:
        csv_fname=tmp_openface_output_folder+'/'+f'{(frame+1):010d}.csv'
        csv_df=pd.read_csv(csv_fname)
        csv_df['frame']=frame
        csv_arr.append(csv_df)
    features_full_df=pd.concat(csv_arr,axis=0)
    # write features to csv not to lose them
    features_full_df.to_csv(output_csv_folder+'/allfeats.csv')

    # output features to json folder
    frames=features_full_df['frame'].to_list()
    headpose=list(zip(features_full_df['pose_Rz'],features_full_df['pose_Ry'],features_full_df['pose_Rx']))
    headpose_dict={'frame':frames,'headpose':headpose}
    json.dump(headpose_dict,open(output_json_folder+'/headpose.json','w'))

    gazeX=features_full_df['gaze_angle_x'].to_list()
    gazeY=features_full_df['gaze_angle_y'].to_list()
    gaze_dict={'frame':frames,'gazeX':gazeX,'gazeY':gazeY}
    json.dump(gaze_dict,open(output_json_folder+'/gaze.json','w'))

    noseX=features_full_df['x_27'].to_list()
    noseY=features_full_df['y_27'].to_list()
    nose_dict={'frame':frames,'noseX':noseX,'noseY':noseY}
    json.dump(nose_dict,open(output_json_folder+'/nose.json','w'))

    # cleanup
    subprocess.run(['rm', '-R', videoframes_folder])
    subprocess.run(['rm', '-R', frames_folder])
    subprocess.run(['rm', '-R', noface_frames_folder])
    subprocess.run(['rm', '-R', tmp_openface_output_folder])

if __name__=='__main__':
    bboxes_file='/media/st4Tb/data/ACEEEGATT/ACE910065/childface_files/ACE910065_faces.json'
    input_video_file='/media/st4Tb/data/ACEEEGATT/ACE910065/inp_videos/ACE910065_clip.avi'
    output_video_folder='/media/st4Tb/projects/eeg-cva-inattention/assets/data/ACE910065/video'
    output_csv_folder = '/media/st4Tb/projects/eeg-cva-inattention/assets/data/ACE910065/csv'
    output_json_folder = '/media/st4Tb/projects/eeg-cva-inattention/assets/data/ACE910065/json'
    temp_root_folder='/home/disaev/tmp'
    main(bboxes_file,input_video_file,output_video_folder,output_csv_folder,output_json_folder,temp_root_folder)