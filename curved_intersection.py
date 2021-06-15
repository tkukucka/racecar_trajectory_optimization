import matplotlib.pyplot as plt
import numpy as np

# function finding intersections between a line segment and an arc
def line_arc_intersection(x_c, y_c, r, attitude_1, attitude_2, p1, p2, plot=False):

    # if turning left
    left = False
    if r < 0:
        left = True
        attitude_1, attitude_2 = attitude_2 + np.pi, attitude_1 + np.pi
        r = -r

    # make sure attitudes are between 0 and 2*pi
    while attitude_1 < 0:
        attitude_1 += 2*np.pi
    while attitude_1 >= 2*np.pi:
        attitude_1 -= 2*np.pi
    while attitude_2 < 0:
        attitude_2 += 2*np.pi
    while attitude_2 >= 2*np.pi:
        attitude_2 -= 2*np.pi

    # Method if track segment is not vertical
    if p1[0] != p2[0]:
        # y = m*x + c parameters for line joining p1 and p2
        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        c = p1[1] - m*p1[0]

        # A B C for the quadratic formula. Equations derived by substituting line eqn into circle eqn and solving for x
        A = 1 + m**2
        B = -2*x_c + 2*m*(c-y_c)
        C = x_c**2 + (c - y_c)**2 - r**2

        # solve equation for intersection points using quadratic formula
        if B**2 - 4*A*C >= 0:
            x_i1 = (-B + (B**2 - 4*A*C)**0.5)/(2*A)
            x_i2 = (-B - (B**2 - 4*A*C)**0.5)/(2*A)

            y_i1 = m*x_i1 + c
            y_i2 = m * x_i2 + c
        else:
            x_i1, x_i2 = None, None

        xl = min(p1[0], p2[0])
        xu = max(p1[0], p2[0])

        # check if intersections are within line segment
        if x_i1 is not None and xl < x_i1 < xu:
            p_i1 = x_i1, y_i1
            a_i1 = np.arctan2(y_i1 - y_c, x_c - x_i1)
            if a_i1 < 0:
                a_i1 += 2 * np.pi
        else:
            p_i1, a_i1 = None, None
        if x_i2 is not None and xl < x_i2 < xu:
            p_i2 = x_i2, y_i2
            a_i2 = np.arctan2(y_i2 - y_c, x_c - x_i2)
            if a_i2 < 0:
                a_i2 += 2 * np.pi
        else:
            p_i2, a_i2 = None, None

    # Method if track segment is vertical
    else:
        B = -2*y_c
        C = y_c**2 + (p1[0]-x_c)**2 - r**2

        if B ** 2 - 4*C >= 0:
            y_i1 = (-B + (B ** 2 - 4*C) ** 0.5) / 2
            y_i2 = (-B - (B ** 2 - 4*C) ** 0.5) / 2
            x_i1, x_i2 = p1[0], p1[0]
        else:
            x_i1, x_i2 = None, None

        yl = min(p1[1], p2[1])
        yu = max(p1[1], p2[1])

        # check if intersections are within line segment
        if x_i1 is not None and yl < y_i1 < yu:
            p_i1 = x_i1, y_i1
            a_i1 = np.arctan2(y_i1 - y_c, x_c - x_i1)
            if a_i1 < 0:
                a_i1 += 2 * np.pi
        else:
            p_i1, a_i1 = None, None
        if x_i2 is not None and yl < x_i2 < yu:
            p_i2 = x_i2, y_i2
            a_i2 = np.arctan2(y_i2 - y_c, x_c - x_i2)
            if a_i2 < 0:
                a_i2 += 2 * np.pi
        else:
            p_i2, a_i2 = None, None

    # Check if intersections are within arc and compute ratio along arc where they occur
    p_i, a_i, r_i = [p_i1, p_i2], [a_i1, a_i2], [0, 0]
    for i in range(2):  # for both points
        if p_i[i] is not None:  # intersection was found
            if attitude_1 > 1.5*np.pi and attitude_2 < 0.5*np.pi:  # check if attitude goes through 2*pi to 0
                if 0 <= a_i[i] <= attitude_2 or attitude_1 <= a_i[i] <= 2*np.pi:  # check if intersection is within arc
                    if a_i[i] < np.pi:  # if intersection is after 0 mark
                        r_i[i] = (a_i[i] - attitude_1 + 2*np.pi)/(attitude_2 - attitude_1 + 2*np.pi)
                    else:
                        r_i[i] = (a_i[i] - attitude_1) / (attitude_2 - attitude_1 + 2 * np.pi)
                else:
                    p_i[i] = None
            elif attitude_2 > 1.5*np.pi and attitude_1 < 0.5*np.pi:  # check if attitude goes through 0 to 2*pi
                if 0 <= a_i[i] <= attitude_1 or attitude_2 <= a_i[i] <= 2*np.pi:  # check if intersection is within arc
                    if a_i[i] < np.pi:  # if intersection is after 0 mark
                        r_i[i] = (a_i[i] - attitude_1 + 2*np.pi)/(attitude_2 - attitude_1 + 2*np.pi)
                    else:
                        r_i[i] = (a_i[i] - attitude_1) / (attitude_2 - attitude_1 + 2 * np.pi)
                else:
                    p_i[i] = None
            else:  # normal situation
                if attitude_1 <= a_i[i] <= attitude_2:  # check if intersection is within arc
                    r_i[i] = (a_i[i] - attitude_1)/(attitude_2 - attitude_1)  # ratio along arc where intercestion occurs
                else:
                    p_i[i] = None

    if plot:
        x_plt, y_plt = [], []
        attitude_plt = attitude_1
        n_plot_pts = 20
        for i in range(n_plot_pts+1):
            x_plt.append(x_c - r*np.cos(attitude_plt))
            y_plt.append(y_c + r*np.sin(attitude_plt))
            attitude_plt += (attitude_2-attitude_1)/n_plot_pts
        plt.plot(x_plt, y_plt, c="b")

        for p in p_i:
            if p is not None:
                plt.scatter(p[0], p[1], c="k")

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="r")
        plt.axis("equal")
        plt.show()

    if left:
        for i in range(len(r_i)):
            r_i[i] = 1 - r_i[i]

    return p_i, r_i

# function finding intersections between car trajectory and track
def track_curved_intersection(x_c, y_c, r, attitude_1, attitude_2, track):
    intersections = []

    X = [track.x_in, track.x_out]
    Y = [track.y_in, track.y_out]

    for boundary in range(2):
        for i in range(len(track.x_in)):
            p_i, r_i = line_arc_intersection(x_c, y_c, r, attitude_1, attitude_2,
                                               (X[boundary][i-1], Y[boundary][i-1]), (X[boundary][i], Y[boundary][i]))
            for j in range(2):
                if p_i[j] is not None:
                    intersections.append([p_i[j], r_i[j]])


    return intersections


if __name__ == "__main__":

    #line_arc_intersection(1, 1, 2, 0, 3, (0, 0), (5, 5), plot=True)
    #line_arc_intersection(x_c, y_c, r, attitude_1, attitude_2, p1, p2, plot=True)
    line_arc_intersection(-54.82290595717705, 33.5608378564954, -24.999999999999996, -9.723446849010202, -10.323339699756099, (-75, 0), (-69, 32), plot=True)
    #line_arc_intersection(-103.84504968099668, 24.38868234449054, -40.0, -0, -0.3, (-70, 32), (-58, 40), plot=True)












