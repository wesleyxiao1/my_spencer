import pandas
import argparse

# Takes in the group detections and formats it for MHT
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")

    return parser.parse_args()

def main():
    args = parseArguments()

    df = pandas.read_csv(args.input_file)

    header = "frame,u,v"
    with open(args.output_file, "w") as file:
        file.write(header + "\n")
    
    with open(args.output_file, "a+") as file:
        for _, row in df.loc[df.frameID > 18002].iterrows():
            x = row.x
            y = row.y
            frame = int(row.frameID)
            out = ','.join([str(frame), str(x), str(y)])
            file.write(out + "\n")

    '''
    # get the bounding boxes and break them down to their four coordinates
    #   output them to a file for openmht in the following order:
    #       topLeft
    #       topRight
    #       botLeft
    #       botRight
    for row in df.iterrows():
        x = row.x
        y = row.y
        h = row.h
        w = row.w
        frame = row.frameID

        topLeft  = (x, y)
        topRight = (x+w, y)
        botLeft  = (x, y+h)
        botRight = (x, y)

        coordinates = [topLeft, topRight, botLeft, botRight]
        with open(args.output_file, "w") as file:
            for c in coordinates:
                out = ','.join([str(frame), str(c[0]), str(c[1])])
                file.write(out)
    '''

if __name__ == "__main__":
    main()