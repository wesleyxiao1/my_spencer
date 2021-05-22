import os, sys, math, time
from collections import deque
from multiprocessing import Lock
import numpy, scipy, scipy.spatial.distance, scipy.misc, scipy.cluster.hierarchy
import pandas
import argparse

# Main callback when new social relations and person tracks are available     
def detectGroups(trackedPersons, groups, timestamp):

    # Associate newly created groups with groups from previous cycle for ID consistency, and generate TrackedGroup instances
    trackedGroups = trackGroups(groups, trackedPersons, timestamp)
    return trackedGroups


### Creates group indices by single-linkage clustering
def cluster(trackCount, trackDistances, threshold):
    # Safety check if there are less than 2 tracks, otherwise clustering fails
    if trackCount == 0:
        return []      
    elif trackCount == 1:
        return [0]       

    # Perform clustering
    linkage = scipy.cluster.hierarchy.linkage(trackDistances, method='single')
    groupIndices = scipy.cluster.hierarchy.fcluster(linkage, threshold, 'distance')
    return groupIndices


# Generates a set of track IDs per group
def createGroups(groupIndices, tracks):
    groups = dict()
    trackIndex = 0
    for groupIndex in groupIndices:
        if not groupIndex in groups:
            groups[groupIndex] = []

        trackId = tracks.iloc[trackIndex].pedID
        groups[groupIndex].append(trackId)
        trackIndex += 1
    return groups


# For remembering recent group ID assignments for a given set of track IDs
class GroupIdAssignment(object):
    def __init__(self, trackIds, groupId, createdAt):
        self.trackIds = set(trackIds)
        self.groupId = groupId
        self.createdAt = createdAt

    def __str__(self):
        return "%s = %d" % (str(list(self.trackIds)), self.groupId)

# For publishing consecutive group IDs, even if internally they are not consecutive
class GroupIdRemapping(object):
    def __init__(self, groupId, publishedGroupId):
        self.originalGroupId = groupId
        self.publishedGroupId = publishedGroupId

def remapGroupId(groupId):
    publishedGroupId = None
    for groupIdRemapping in trackGroups.groupIdRemapping:
        if groupIdRemapping.originalGroupId == groupId:
            publishedGroupId = groupIdRemapping.publishedGroupId
            break
    if publishedGroupId is None:
        trackGroups.largestPublishedGroupId += 1
        publishedGroupId = trackGroups.largestPublishedGroupId

    trackGroups.groupIdRemapping.append( GroupIdRemapping(groupId, publishedGroupId) )
    return publishedGroupId

class BoundingBox():
    def __init__(self, x, y, height, width):
        self.x = x
        self.y = y
        self.height = height
        self.width = width

class TrackedGroup():
    def __init__(self, groupID, pedIDs, x, y):
        self.groupID = groupID
        self.pedIDs = pedIDs
        self.x = x
        self.y = y
        self.bbox = None

    def __str__(self):
        out = [str(self.groupID), str(self.x), str(self.y)]
        return ",".join(out)

class GroupCentroid():
    def __init__(self, pos=0, groupId=0, size=0):
        self.pos = pos
        self.groupId = groupId
        self.size = size

### Calculates group centroids for the current groups
def calculateGroupCentroids(groups, trackedPersons):
    # Get track positions
    trackPositionsById = dict()
    for track in trackedPersons.tracks:
        pos = track.pose.pose.position            
        trackPositionsById[track.track_id] = [ pos.x, pos.y ]

    # Determine centroids
    centroids = dict()
    for groupId, track_ids in groups.iteritems():
        positions = numpy.zeros( (len(track_ids), 2) )
        trackIndex = 0
        for track_id in track_ids:
            positions[trackIndex, :] = trackPositionsById[track_id]
            trackIndex += 1

        currentCentroid = GroupCentroid()
        currentCentroid.pos = numpy.mean(positions, 0)
        currentCentroid.groupId = groupId
        currentCentroid.size = len(track_ids)
        centroids[groupId] = currentCentroid
    return centroids

# Associates current groups with previously tracked groups via track IDs
def trackGroups(groups, trackedPersons, timestamp):
    # Initialize variables
    publishSinglePersonGroups = False   # add to parameters of this script
    trackedGroups = []
    assignedGroupIds = []

    # Sort groups by smallest track ID per group to ensure reproducible group ID assignments
    #sortedGroups = sorted(groups.iteritems(), key=lambda (clusterId, track_ids) : sorted(track_ids)[0])
    #import pdb; pdb.set_trace()
    sortedGroups = sorted(groups.items(), key=lambda group : sorted(group[1])[0])

    # Used to calculate group centroids
    trackPositionsById = dict()
    for _, trackedPerson in trackedPersons.iterrows():           
        trackPositionsById[trackedPerson.pedID] = numpy.array([ trackedPerson.x, trackedPerson.y ])

    # Create a TrackedGroup for each group, and assign a unique ID
    for clusterId, track_ids in sortedGroups:            
        # Check if we encountered this combination of track IDs, or a superset thereof, before
        bestGroupIdAssignment = None

        trackIdSet = set(track_ids)
        for groupIdAssignment in trackGroups.groupIdAssignmentMemory:
            if groupIdAssignment.trackIds.issuperset(trackIdSet) or groupIdAssignment.trackIds.issubset(trackIdSet):
                trackCount = len(groupIdAssignment.trackIds)
                bestTrackCount = None if bestGroupIdAssignment is None else len(bestGroupIdAssignment.trackIds)

                if bestGroupIdAssignment is None or trackCount > bestTrackCount or (trackCount == bestTrackCount and groupIdAssignment.createdAt < bestGroupIdAssignment.createdAt):
                    if groupIdAssignment.groupId not in assignedGroupIds:
                        bestGroupIdAssignment = groupIdAssignment

        groupId = None
        if bestGroupIdAssignment is not None:
            groupId = bestGroupIdAssignment.groupId

        if groupId == None or groupId in assignedGroupIds:
            groupId = trackGroups.largestGroupId + 1 # just generate a new ID

        # Remember that this group ID has been used in this cycle
        assignedGroupIds.append(groupId)

        # Remember which group ID we assigned to this combination of track IDs
        groupIdAssignmentsToRemove = []
        groupExistsSince = timestamp
        for groupIdAssignment in trackGroups.groupIdAssignmentMemory:
            if set(track_ids) == groupIdAssignment.trackIds:
                groupExistsSince = min(groupIdAssignment.createdAt, groupExistsSince)
                groupIdAssignmentsToRemove.append(groupIdAssignment)  # remove any old entries with same track IDs
        
        for groupIdAssignment in groupIdAssignmentsToRemove:
            trackGroups.groupIdAssignmentMemory.remove(groupIdAssignment) 

        trackGroups.groupIdAssignmentMemory.append( GroupIdAssignment(track_ids, groupId, groupExistsSince) )
        
        # Remember largest group ID used so far
        if(groupId > trackGroups.largestGroupId):
            trackGroups.largestGroupId = groupId

        # Do not publish single-person groups if not desired
        if publishSinglePersonGroups or len(track_ids) > 1:
            accumulatedPosition = numpy.array([0.0, 0.0])
            activeTrackCount = 0
            for track_id in track_ids:
                if track_id in trackPositionsById:
                    accumulatedPosition += trackPositionsById[track_id]
                    activeTrackCount += 1
        
            trackedGroup = TrackedGroup(groupID=remapGroupId(groupId), 
                                        pedIDs=track_ids,
                                        x=accumulatedPosition[0] / float(activeTrackCount),
                                        y=accumulatedPosition[1] / float(activeTrackCount))
            trackedGroups.append(trackedGroup)

    # Return currently tracked groups
    return trackedGroups

def calcBoundingBoxes(trackedGroups, trackedPersons):
    #import pdb; pdb.set_trace()
    for group in trackedGroups:
        xs = []
        ys = []
        for ped in group.pedIDs:
            currPed = trackedPersons.loc[trackedPersons.pedID == ped]
            x = currPed.iloc[0].x
            y = currPed.iloc[0].y
            w = currPed.iloc[0].w
            h = currPed.iloc[0].h
            
            xs.extend([x, x+w])
            ys.extend([y, y+h])

        X = min(xs)
        Y = min(ys)
        height = max(ys) - min(ys)
        width = max(xs) - min(xs)
        group.bbox = BoundingBox(round(X,3), round(Y,3), round(height,3), round(width,3))

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--social_relations")
    parser.add_argument("--group_detections")
    parser.add_argument("--output_file")

    return parser.parse_args()

### Main method
def main():
    args = parseArguments()

    df_tracking = pandas.read_csv(args.input_file)
    df_tracking = df_tracking.loc[(df_tracking.dataset == 'test')]

    header = "frameID,groupID,x,y,w,h"
    with open(args.output_file, "w") as file:
        file.write(header + "\n")

    trackGroups.largestGroupId = -1
    trackGroups.groupIdAssignmentMemory = deque(maxlen=300)
    trackGroups.groupIdRemapping = deque(maxlen=50)
    trackGroups.largestPublishedGroupId = -1

    # detect groups frame by frame
    frames = df_tracking.frameID.unique()
    timestamp = 0

    group_detections_file = args.group_detections
    with open(group_detections_file, "r") as file:
        groups = dict()
        group_index = 0
        current_frame = -1
        for line in file.readlines():
            content = line.split(",")
            frame = int(content[0])
            if current_frame == -1:
                current_frame = frame

            if current_frame != frame:
                #print("current frame = ", current_frame)
                trackedPersons = df_tracking.loc[(df_tracking.frameID == current_frame)]
                if len(trackedPersons) == 0:
                    groups = dict()
                    group_index = 0
                    timestamp += 1
                    current_frame = frame
                    if current_frame % 1000 == 0:
                        from datetime import datetime
                        now = datetime.now().time() # time object
                        print(current_frame, now)
                    continue
                
                trackedGroups = detectGroups(trackedPersons, groups, timestamp)
                '''
                print(len(trackedGroups))
                if len(trackedGroups) != 0:
                    print(trackedGroups[0])
                    print(trackedGroups[0].groupID)
                    print(trackedGroups[0].pedIDs)
                    print(trackedPersons)
                '''

                calcBoundingBoxes(trackedGroups, trackedPersons)
                with open(args.output_file, "a+") as file:
                    for g in trackedGroups:
                        out = ','.join([str(current_frame), 
                                        str(g.groupID), 
                                        str(g.bbox.x), 
                                        str(g.bbox.y), 
                                        str(g.bbox.width),
                                        str(g.bbox.height)])
                        file.write(out + "\n")
                current_frame = frame
                groups = dict()
                group_index = 0
                timestamp += 1

                if current_frame % 1000 == 0:
                    from datetime import datetime
                    now = datetime.now().time() # time object
                    print(current_frame, now)

            pedIDs = [int(c) for c in content[1:]]

            groups[group_index] = pedIDs
            group_index += 1

### Entry point
if __name__ == '__main__':
    main()