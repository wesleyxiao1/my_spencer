import os, sys, math, time
from collections import deque
from multiprocessing import Lock
import numpy, scipy, scipy.spatial.distance, scipy.misc, scipy.cluster.hierarchy
import pandas
import argparse

# Main callback when new social relations and person tracks are available     
def detectGroups(trackedPersons, socialRelations, timestamp):
    # Collect track positions in matrix (rows = tracks, cols = x and y coordinates)
    #trackCount = len(trackedPersons.pedID.unique())
    trackCount = len(trackedPersons)

    # Create a lookup from arbitrary track IDs to zero-based track indices in distances matrix
    trackIndex = 0
    trackIdToIndex = dict()
    for _, trackedPerson in trackedPersons.iterrows():
        trackIdToIndex[trackedPerson.pedID] = trackIndex
        trackIndex += 1

    # Build full (symmetric) distances matrix from provided person relations. Diagonal is initialized with zeroes.
    socialRelationsMatrix = numpy.ones( (trackCount, trackCount) )
    for _, socialRelation in socialRelations.iterrows():
        try:
            index1 = trackIdToIndex[socialRelation.trackID1]
            index2 = trackIdToIndex[socialRelation.trackID2]
        except KeyError:
            print("Key error while looking up tracks %d and %d!" % (socialRelation.trackID1, socialRelation.trackID2) )
            continue
            
        # Using min() here in case there are multiple types of relations per track pair -- in that case, use strongest type of relation
        socialRelationsMatrix[index1, index2] = min(socialRelationsMatrix[index1, index2], 1.0 - socialRelation.strength) # strong relation --> small distance
        socialRelationsMatrix[index2, index1] = min(socialRelationsMatrix[index2, index1], 1.0 - socialRelation.strength)

    for i in range(0, trackCount):
        socialRelationsMatrix[i,i] = 0.0 # diagonal elements have to be zero

    # Convert into condensed form
    trackDistances = scipy.spatial.distance.squareform(socialRelationsMatrix, force='tovector')
        
    # Cluster
    relationThreshold = 0.75 # relation strength above which we consider tracks to be in a group
    groupIndices = cluster(trackCount, trackDistances, relationThreshold) # outputs one group index per track

    #import pdb; pdb.set_trace()
    # Create groups, by assembling one set of track IDs per group
    groups = createGroups(groupIndices, trackedPersons)

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
    for group in trackedGroups:
        X, Y = math.inf, math.inf
        height = 0
        width = 0
        for ped in group.pedIDs:
            currPed = trackedPersons.loc[trackedPersons.pedID == ped]
            x = currPed.iloc[0].x
            y = currPed.iloc[0].y
            w = currPed.iloc[0].w
            h = currPed.iloc[0].h

            X = min(X, x)
            Y = min(Y, y)
            height = max(height, h)
            width = max(width, w)
        group.bbox = BoundingBox(X, Y, height, width)

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--social_relations")
    parser.add_argument("--output_file")

    return parser.parse_args()

### Main method
def main():
    args = parseArguments()

    df_social_relations = pandas.read_csv(args.social_relations)
    df_tracking = pandas.read_csv(args.input_file)

    df_tracking = df_tracking.loc[((df_tracking.dataset == 'test') & (df_tracking.segment_num == 1))]

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
    '''
    n = 4
    for f in frames[::n]:
        f_end = f + n - 1
        trackedPersons = df_tracking.loc[((df_tracking.frameID >= f) & (df_tracking.frameID <= f_end))]
        socialRelations = df_social_relations.loc[((df_social_relations.frameID >= f) & (df_social_relations.frameID <= f_end))]
    '''
    for f in frames:
        trackedPersons = df_tracking.loc[(df_tracking.frameID == f)]
        socialRelations = df_social_relations.loc[(df_social_relations.frameID == f)]
        # get groups in frame
        groups = detectGroups(trackedPersons, socialRelations, timestamp)
        calcBoundingBoxes(groups, trackedPersons)

        # write groups to outfile
        with open(args.output_file, "a+") as file:
            for g in groups:
                out = ','.join([str(f), 
                                str(g.groupID), 
                                str(g.x), 
                                str(g.y), 
                                str(g.bbox.width),
                                str(g.bbox.height)])
                file.write(out + "\n")

        if f % 100 == 0:
            from datetime import datetime
            now = datetime.now().time() # time object
            print(f,now)
        timestamp += 1

### Entry point
if __name__ == '__main__':
    main()