import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from curved_intersection import track_curved_intersection
from straight_intersection import track_straight_intersection

colormap = plt.get_cmap("jet")

# Class defining the race track
class Track:
    def __init__(self):
        x = [-51.6, -46.6, -16.0, 24.4, 56.8, 59.6, 45.8, 46.2, 13.8, -2.2, -34.8, -75.2, -69, -22.4, 26.4, 72.4, 76.6, 66.4, 61.8, 5.2, -11.4, -46.8]
        y = [-0.4, 22.4, 47.6, 50.8, 46.4, 23.0, -3.8, -33.4, -45.2, -19.8, -22.8, 0.0, 32.2, 70.6, 69.2, 59.2, 18.6, -7.6, -50.6, -58.4, -33.6, -35.0]

        n = int(len(x)/2)
        self.x_in = x[:n]
        self.x_out = x[n:]
        self.y_in = y[:n]
        self.y_out = y[n:]

    def plot(self):
        plt.plot(self.x_in+[self.x_in[0]], self.y_in+[self.y_in[0]], c="k")
        plt.plot(self.x_out+[self.x_out[0]], self.y_out+[self.y_out[0]], c="k")


# Function to compute new velocity after a time interval dt
def velocity_function(v_1, gas, dt):  # both velocity and gas in range from 0 to 1, dt must not be larger than 1
    mass_constant = 0.25
    return v_1 + (gas - v_1)*mass_constant*dt

# used to assign color for different kinds of plot
def color_function(color, style, inside_plt):
    if style == None:
        # plotting inside track blue and outside red
        color = "r"
        if inside_plt == 1:
            color = "b"
        # color define manually
    if style in ["r", "g", "b"]:
        color = style
        style = "-"
    return color, style

def next_pos(x_1, y_1, attitude_1, steer_angle, velocity, dt, track, inside, penalty_time, plot=True, style="-"):
    distance = velocity * dt

    arc = abs(steer_angle) > 1e-14

    # if trajectory is an arc
    if arc:
        r = 1/steer_angle  # compute turn radius
        x_c = x_1 + r*np.cos(attitude_1)  # x_c and y_c are the coordinates of the center of circle forming the arc
        y_c = y_1 - r*np.sin(attitude_1)
        turn_angle = distance*steer_angle  # angle of the arc traced out
        attitude_2 = attitude_1 + turn_angle  # direction car faces at end of turn
        x_2 = x_c - r*np.cos(attitude_2)  # x_2 and y_2 is the new position
        y_2 = y_c + r*np.sin(attitude_2)

        # find intersection points between trajectory and track
        intersections = track_curved_intersection(x_c, y_c, r, attitude_1, attitude_2, track)

    # if trajectory is a straight line
    else:  # different procedure for straight segments to avoid division by zero
        attitude_2 = attitude_1  # new attitude is the same
        x_2 = x_1 + distance*np.sin(attitude_1)  # compute new position of car
        y_2 = y_1 + distance*np.cos(attitude_1)

        # find intersection points between trajectory and track
        intersections = track_straight_intersection((x_1, y_1), (x_2, y_2), track)

    # plotting trajectory

    color, style = color_function("r", style, True)
    if style == "v":
        v_max, v_min = 15, 0
        frac = (velocity - v_max)/(v_min - v_max)
        color = colormap(1-frac)
        linestyle = "-"
    else:
        linestyle = style

    if plot:
        inside_plt = inside
        #if style is None or style is "v":
        plt.scatter([x_1, x_2], [y_1, y_2], c="k")
        if arc:
            x_plt_0, y_plt_0 = x_1, y_1
            attitude_plt = attitude_1
            if style == ":":
                n_plot_pts = 5
            else:
                n_plot_pts = 50

            for i in range(n_plot_pts):
                last_attitude = attitude_plt
                attitude_plt += turn_angle/n_plot_pts

                # check if intersection occurs between plot segments of curve segment
                if len(intersections) > 0:
                    for j in range(len(intersections)):
                        r_i = intersections[j][1]
                        if (last_attitude - attitude_1)/turn_angle <= r_i < (attitude_plt - attitude_1)/turn_angle:
                            inside_plt *= -1

                color, style = color_function(color, style, inside_plt)

                x_plt_1 = x_c - r*np.cos(attitude_plt)
                y_plt_1 = y_c + r*np.sin(attitude_plt)
                plt.plot([x_plt_0, x_plt_1], [y_plt_0, y_plt_1], c=color, linestyle=linestyle)
                x_plt_0, y_plt_0 = x_plt_1, y_plt_1
        else:
            inside_plt = inside

            color, style = color_function(color, style, inside_plt)

            if len(intersections) > 0:
                plt.plot([x_1, intersections[0][0][0]], [y_1, intersections[0][0][1]], c=color, linestyle=linestyle)

                if len(intersections) > 1:
                    for j in range(len(intersections)-1):
                        inside_plt *= -1

                        color, style = color_function(color, style, inside_plt)
                        plt.plot([intersections[j][0][0], intersections[j+1][0][0]],
                                 [intersections[j][0][1], intersections[j+1][0][1]], c=color, linestyle=linestyle)

                inside_plt *= -1
                color, style = color_function(color, style, inside_plt)

                plt.plot([intersections[-1][0][0], x_2], [intersections[-1][0][1], y_2], c=color, linestyle=linestyle)
            else:
                plt.plot([x_1, x_2], [y_1, y_2], c=color, linestyle=linestyle)

        for p in intersections:
            pass
            #plt.scatter(p[0][0], p[0][1], c="r")

    # Compute penalty time and inside parameter
    if len(intersections) == 0:  # if no intersections present
        if inside == -1:  # whole segment outside of track
            penalty_time += dt

    # if intersections present
    else:
        # create list of segments within curve separated by intersections
        r_list = [0]
        for j in range(len(intersections)):
            r_list.append(intersections[j][1])
        r_list.append(1)
        r_list.sort()  # in case intersections are in wrong order

        for j in range(len(r_list)-1):
            if inside == -1:
                penalty_time += dt*(r_list[j+1] - r_list[j])
            inside *= -1
        inside *= -1  # multiply once more to fix for end of curve

    # compute centripetal acceleration
    if arc:
        a_c = velocity**2/abs(r)
    else:
        a_c = 0

    return x_2, y_2, attitude_2, inside, penalty_time, a_c

# Function for integrating the steering vector and gas vector to get the trajectory, returns the number of laps
def integrate_position(design_vector, track, dt=1, a_max=1, plot=False, print_res=False, style="-"):

    # slice design vector
    n = int(len(design_vector)/2)
    steering_vector = design_vector[:n]
    gas_vector = design_vector[n:]

    # initializing variables
    max_velocity = 15
    max_steer_angle = 0.25

    x, y = -60, 0
    velocity = 0
    attitude = 0
    inside = 1  # 1 = inside track, -1 = outside track
    penalty_time, penalty_acceleration = 0, 0
    angle_covered = 0
    theta = np.arctan2(y, -x)

    for i in range(n):
        # update position
        velocity = velocity_function(velocity, gas_vector[i], dt)

        x, y, attitude, inside, penalty_time, a_c = \
            next_pos(x, y, attitude, steering_vector[i]*max_steer_angle, velocity*max_velocity, dt, track,
                     inside, penalty_time, plot=plot, style=style)

        # update angle covered
        theta_new = np.arctan2(y, -x)
        if theta_new < 0:  # make theta be between 0 - 2*pi
            theta_new += 2*np.pi

        if theta_new < 0.5*np.pi and theta > 1.5*np.pi:  # check if theta passes through 0
            angle_covered += theta_new - theta + 2*np.pi
        elif theta < 0.5*np.pi and theta_new > 1.5*np.pi:  # check if theta passes through 0 backwards
            angle_covered += theta_new - theta - 2*np.pi
        else:
            angle_covered += theta_new - theta
        theta = theta_new

        # update acceleration penalty
        if a_c > a_max:
            penalty_acceleration += (a_c/a_max-1)*dt


    n_laps = angle_covered/(2*np.pi)

    # print computation time
    if print_res:
        print("Number of laps:", n_laps)
        print("Penalty time:", penalty_time)
        print("Penalty acceleraton", penalty_acceleration)

    # plot track
    if plot:
        track.plot()
        plt.axis("equal")
        #plt.show()

    return n_laps, penalty_time, penalty_acceleration

# Objective function
def objective_1(x, args, plot=False, style="-"):
    s_i, g_i, dt, a_max, gas, constraints = args

    if plot and style == "v":
        plt.colorbar(ScalarMappable(cmap=colormap), label=r"$V/V_{max}$", fraction=0.05)


    p_g, p_s = 0, 0
    # design vector setup for optimizing both steering and velocity
    if gas:
        n = int(len(x)/2)
        s_vec = s_i + list(x[:n])
        g_vec = g_i + list(x[n:])

        # compute penalty for gas out of bounds
        for g in g_vec:
            if g > 1:
                p_g += (g - 1)

    # design vector setup for optimizing steering only
    else:
        s_vec = s_i + list(x)
        g_vec = [1] * len(s_vec)

    for s in s_vec:
        if abs(s) > 0.5:
            p_s += abs(s) - 0.5

    n_laps, p_t, p_a = integrate_position(s_vec+g_vec, Track(), a_max=a_max, dt=dt, plot=plot, print_res=plot, style=style)
    if constraints:  # return basic objective
        obj = -n_laps

    else:  # return objective with penalty function
        #obj = -n_laps + p_t #+ 0.1*p_a + p_g
        obj = -n_laps + p_t**2*1.5 + p_a**2 + p_g + p_s

    if plot:
        print("Penalty gas:", p_g)
    return obj

# unfortunately scipy.minimize does not call objective and constraint functions in a consistent order, therefore
# constraints cant be saved, instead the model has to be rerun
def constraints_1(x, args, plot=False, style="-"):
    s_i, g_i, dt, a_max, gas, constraints = args

    p_g = 0
    # design vector setup for optimizing both steering and velocity
    if gas:
        n = int(len(x) / 2)
        s_vec = s_i + list(x[:n])
        g_vec = g_i + list(x[n:])

        # compute penalty for gas out of bounds
        for g in g_vec:
            if g > 1:
                p_g += (g - 1)

    # design vector setup for optimizing steering only
    else:
        s_vec = s_i + list(x)
        g_vec = [1] * len(s_vec)

    n_laps, p_t, p_a = integrate_position(s_vec+g_vec, Track(), dt=dt, plot=plot, print_res=plot, style=style, a_max=a_max)
    #out = -p_t**2, -p_a**2, p_g
    out = -p_t, -p_a, -p_g

    return out

if __name__ == "__main__":
    s_vec = [0, 0.2, 0.1, 0, 0, 0, 0.2, 0.1, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.4, 0, -0.4, 0, 0.2, 0.2, 0.1]
    g_vec = [1]*len(s_vec)

    s_vec = []
    for i in range(len(g_vec)):
        s_vec.append(random.randint(-50, 50)/100)
        if -0.3 < s_vec[i] < 0.3:
            s_vec[i] = 0
    print(s_vec)

    s_vec = [-0.04590722, 0.2, 0.1, 0, 0, 0, 0.2, 0.1, 0, 0, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.4, 0, -0.4, 0, 0.2,
             0.2, 0.03956485]

    out, penalty_time, penalty_acceleration = integrate_position(s_vec+g_vec, Track(), print_res=True, plot=True)


