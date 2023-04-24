import json
from pathlib import Path
import subprocess
import os
import pandas as pd

OPENFACE_EXEC = '<path to openface>/OpenFace/build/bin/FaceLandmarkImg'
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
   
    tmp_openface_output_folder='<path to temporary openface output folder>'
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
    bboxes_file='<path to json file with bounding boxes>'
    input_video_file='<path to video file>'
    output_video_folder='<output folder for video>'
    output_csv_folder = '<output folder for csv>'
    output_json_folder = '<output folder for jsons>'
    temp_root_folder='<path to tmp folder>'
    main(bboxes_file,input_video_file,output_video_folder,output_csv_folder,output_json_folder,temp_root_folder)