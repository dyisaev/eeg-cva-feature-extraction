from FaceTrackerSession import FaceTrackerSession
from argparse import ArgumentParser

parser = ArgumentParser(
                    prog = 'face_detection',
                    description = 'Detector of head bounding boxes with ambiguity resolution',
                    epilog = 'Duke University, Sapiro Lab, 2022')
parser.add_argument('-u','--user',required=True)
parser.add_argument('-l','--log',required=True)
parser.add_argument('-i','--input-video',required=True)
parser.add_argument('-o','--output-video',required=True)
parser.add_argument('-j','--output-json',required=True)

def main(username,logfile,input_video,output_video,output_json):
    ftSession = FaceTrackerSession(username,logfile)
    ftSession.process_video(input_video,output_video,output_json)
    
if __name__=='__main__':
    args = parser.parse_args()
    print(args)
    main(args.user,args.log,args.input_video,args.output_video,args.output_json)
