import pandas as pd
import rospy
import std_msgs.msg
from geometry_msgs.msg import Point
from spencer_trcking_msgs.msg import TrackedPersons, TrackedPerson

INPUT_FILE = 'data/group_track_Data.csv'

def main():
    rospy.init_node("tracked_peds")
    trackedPersonsTopic = rospy.resolve_name("/spencer/perception/tracked_persons")
    pub = rospy.Publisher(trackedPersonsTopic, trackedPersons)

    df_ped_tracks = pd.read_csv(INPUT_FILE)
    frames = df_ped_tracks['frameID'].unique()

    for f in frames:
        peds = df_ped_tracks[df_ped_tracks['frameID'] == f]

        ped_tracks = []
        for _, p in peds.interrow():
            msg = TrackedPerson()
            msg.track_id = p['pedID']

            position = Point(p['x'], p['y'], 0)
            msg.pose.pose.position = position
            ped_tracks.append(msg)
        
        msg = TrackedPersons()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        msg.header = header
        msg.tracks = ped_tracks
        pub.publish(msg)

if __name__ == '__main__':
    main()

