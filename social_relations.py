import math
import pandas
from sklearn import svm
import joblib
import numpy as np
import argparse

maxDistance = 3.0
maxDeltaSpeed = 1.0
maxDeltaAngle = math.pi / 4
minSpeedToConsiderAngle = 0.1

class SocialRelation:
    def __init__(self, strength, track1_id, track2_id, frame_id):
        self.strength = strength
        self.track1_id = track1_id
        self.track2_id = track2_id
        self.frame_id = frame_id

def distance(det1, det2):
    return math.hypot(det1.iloc[0].x - det2.iloc[0].x, det1.iloc[0].y - det2.iloc[0].y)

def speed(det, df, n):
    # go back n frames to calc speed
    curr_frameID = det.iloc[0].frameID
    curr_pedID = det.iloc[0].pedID

    # check previous frames and that they exist
    prev_dets = df.loc[(df.pedID == curr_pedID) & (df.frameID < curr_frameID)]
    if len(prev_dets) == 0:
        return 0
    
    # go back n frames (or as far back as possible) to calc distance
    prev_frame = min(n, len(prev_dets))
    prev_det = None
    while prev_frame >= 0:
        prev_det = df.loc[(df.pedID == curr_pedID) & (df.frameID == curr_frameID - prev_frame)]
        if len(prev_det) == 0:
            prev_frame -= 1
        else:
            break
    if len(prev_det) == 0:
        return 0

    dist = distance(det, prev_det)
    return dist / n

def delta_speed(det1, det2, df, n):
    speed1 = speed(det1, df, n)
    speed2 = speed(det2, df, n)
    return speed1 - speed2

def delta_angle(det1, det2):
    vector1 = np.array([det1.iloc[0].x, det1.iloc[0].y])
    vector2 = np.array([det2.iloc[0].x, det2.iloc[0].y])

    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle

def newTrackedPersonsReceived(trackedPersons, n, output_file):
    model = joblib.load('social_relations_model/data/social_relations_penalty_1_unbalanced.model')

    socialRelations = []
    frames = trackedPersons.frameID.unique()
    for current_frame in frames:
        trackCount = trackedPersons.loc[trackedPersons.frameID == current_frame].pedID.unique()
        for t1_index in trackCount:
            for t2_index in trackCount:
                if t1_index == t2_index:
                    continue
                t1 = trackedPersons.loc[(trackedPersons.frameID == current_frame) & \
                                        (trackedPersons.pedID == t1_index)]
                t2 = trackedPersons.loc[(trackedPersons.frameID == current_frame) & \
                                        (trackedPersons.pedID == t2_index)]
                #import pdb; pdb.set_trace()
                # Calculate final feature values
                dist = distance(t1, t2)
                speed = delta_speed(t1, t2, trackedPersons, n)
                angle = delta_angle(t1, t2)
                
                # Gating for large distance, very different velocities, or very different angle
                '''
                if dist > maxDistance or speed > maxDeltaSpeed or angle > maxDeltaAngle:
                    positiveRelationProbability = 0.1
                    negativeRelationProbability = 0.9
                else:
                    # Prepare SVM classifier
                    vector = np.array([[dist, speed, angle]])

                    # Run SVM classifier
                    positiveRelationProbability = model.predict_proba(vector)[:,1]
                    negativeRelationProbability = 1 - positiveRelationProbability
                '''
                vector = np.array([[dist, speed, angle]])

                # Run SVM classifier
                positiveRelationProbability = model.predict_proba(vector)[:,1]
                negativeRelationProbability = 1 - positiveRelationProbability

                if positiveRelationProbability < 0.1:
                    positiveRelationProbability = 0.1
                    negativeRelationProbability = 0.9

                # Store results for this pair of tracks
                if isinstance(positiveRelationProbability, float):
                    socialRelation = SocialRelation(positiveRelationProbability, t1.iloc[0].pedID, t2.iloc[0].pedID, current_frame)
                else:
                    socialRelation = SocialRelation(positiveRelationProbability[0], t1.iloc[0].pedID, t2.iloc[0].pedID, current_frame)    
                    
                socialRelations.append(socialRelation)
        if current_frame % 100 == 0:
            from datetime import datetime
            now = datetime.now().time() # time object
            print(current_frame,now)

        with open(output_file, "a+") as outfile:
            for s in socialRelations:
                outfile.write(str(s.frame_id) + "," + str(s.strength) + "," + str(s.track1_id) + "," + str(s.track2_id) + "\n")
            socialRelations = []
    return socialRelations

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    parser.add_argument("--num_frames_for_speed", help="number of frames to go back when calculating the speed of a pedestrian", type=int, default=1)

    return parser.parse_args()

def main():
    args = parseArguments()

    header = "frameID,strength,trackID1,trackID2\n"
    with open(args.output_file, "w") as outfile:
        outfile.write(header)

    df = pandas.read_csv(args.input_file)
    #data = df.loc[df.dataset == 'test']
    data = df
    socialRelations = newTrackedPersonsReceived(data, args.num_frames_for_speed, args.output_file)

    '''
    header = "frameID,strength,trackID1,trackID2\n"
    with open(args.output_file, "w") as outfile:
        outfile.write(header)
        for s in socialRelations:
            print("frame id   = " + str(s.frame_id))
            print("strength   = " + str(s.strength))
            print("track id 1 = " + str(s.track1_id))
            print("track id 2 = " + str(s.track2_id))
            outfile.write(str(s.frame_id) + "," + str(s.strength) + "," + str(s.track1_id) + "," + str(s.track2_id) + "\n")
    '''

if __name__ == "__main__":
    main()