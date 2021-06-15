import matplotlib.pyplot as plt

def line_line_intersection(p_1, p_2, q_1, q_2, plot=False):

    intersection, ratio = None, None

    # check line p is not vertical
    if p_1[0] != p_2[0]:
        # make line equation from points p
        mp = (p_2[1] - p_1[1]) / (p_2[0] - p_1[0])
        cp = p_1[1] - mp * p_1[0]
        vp = False
    else:
        vp = True  # line is vertical

    # check line q is not vertical
    if q_1[0] != q_2[0]:
        # make line equation from points q
        mq = (q_2[1] - q_1[1]) / (q_2[0] - q_1[0])
        cq = q_1[1] - mq * q_1[0]
        vq = False
    else:
        vq = True  # line is vertical

    if not (vq or vp):  # neither line is vertical
        if mp != mq:  # check than lines are not parallel
            x_i = (cq - cp)/(mp - mq)  # equation for intersection

            # check if intersection lies within line segments
            if min(p_1[0], p_2[0]) <= x_i <= max(p_1[0], p_2[0]):
                if min(q_1[0], q_2[0]) <= x_i <= max(q_1[0], q_2[0]):
                    intersection = x_i, mp*x_i + cp
                    ratio = (x_i - p_1[0])/(p_2[0] - p_1[0])

    # if p is vertical
    elif (vp or vq) and not (vp and vq):
        if vp:
            y_i = mq*p_1[0] + cq  # equation for intersection
        else:
            y_i = mp * q_1[0] + cp
        if min(q_1[1], q_2[1]) <= y_i <= max(q_1[1], q_2[1]):  # check that intersection lies within line segments
            if min(p_1[1], p_2[1]) <= y_i <= max(p_1[1], p_2[1]):
                if vp:
                    intersection = p_1[0], y_i
                else:
                    intersection = q_1[0], y_i
                ratio = (y_i - p_1[1]) / (p_2[1] - p_1[1])

    if plot:
        plt.plot([p_1[0], p_2[0]],[p_1[1], p_2[1]])
        plt.plot([q_1[0], q_2[0]], [q_1[1], q_2[1]])
        if intersection is not None:
            plt.scatter(intersection[0], intersection[1])
        print(intersection)
        plt.show()

    return intersection, ratio

# function to find intersections between any track segment and a straight trajectory segment
def track_straight_intersection(p_1, p_2, track):
    intersections = []

    X = [track.x_in, track.x_out]
    Y = [track.y_in, track.y_out]

    for boundary in range(2):
        for i in range(len(track.x_in)):
            p_i, r_i = line_line_intersection(p_1, p_2, (X[boundary][i-1], Y[boundary][i-1]), (X[boundary][i], Y[boundary][i]))
            if p_i is not None:
                intersections.append([p_i, r_i])

    return intersections


if __name__=="__main__":
    line_line_intersection((-2, 3), (1, 2), (-1, 4), (-1, 2), plot=True)






