import torch
import numpy as np
from .conversions import quat_to_rotMat
from .data_utils import XSensDataIndices
from numpy.linalg import norm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

class Skeleton:

    def __init__(self):
        segments = ['Pelvis', 'L5', 'L3', 
                    'T12', 'T8', 'Neck', 'Head',
                    'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                    'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                    'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe',
                    'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot', 'LeftToe']

        parents = [None, 'Pelvis', 'L5', 
                   'L3', 'T12', 'T8', 'Neck',
                   'T8', 'RightShoulder', 'RightUpperArm', 'RightForeArm',
                   'T8', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm',
                   'Pelvis', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot',
                   'Pelvis', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot'] 

        body_frame_positions = [None, [0.00000, 0.000000, 0.103107], [0.000000, 0.000000, 0.095793], 
                                [0.00000, 0.000000, 0.087564], [0.000000, 0.000000, 0.120823], [0.00000, 0.000000, 0.103068], [0.000000, 0.000000, 0.208570],
                                [0.000000, -0.027320, 0.068158], [0.000000, -0.141007, 0.000000], [0.000000, -0.291763, 0.000000], [0.000071, -0.240367, 0.000000],
                                [0.000000, 0.027320, 0.068158], [0.000000, 0.141007, 0.000000], [0.000000, 0.291763, 0.000000], [ 0.000071, 0.240367, 0.000000],
                                [0.000, -0.087677, 0.000], [-0.000055, 0.000000, -0.439960], [0.000248, 0.000000, -0.445123], [0.192542, 0.000000, -0.087304], 
                                [0.000, 0.087677, 0.000], [-0.000055, 0.000000, -0.439960], [0.000248, 0.000000, -0.445123], [0.192542, 0.000000, -0.087304]]         
        
        self.skeleton_tree = [[0, 1, 2, 3, 4],
                              [4, 7, 8, 9, 10],
                              [4, 11, 12, 13, 14],
                              [4, 5, 6],
                              [0, 15, 16, 17, 18],
                              [0, 19, 20, 21, 22]]

        self.segments = segments
        self.index_of = dict(zip(segments, range(len(segments))))
        self.segment_parents = dict(zip(segments, parents))
        self.segment_positions_in_parent_frame = dict(zip(segments, body_frame_positions))

    def forward_kinematics(self, orientations):
        xsens_indices = XSensDataIndices()

        positions = torch.zeros([len(self.segments), orientations.shape[0], 3], dtype=torch.float64)   
 
        for i, segment in enumerate(self.segments):
            parent = self.segment_parents[segment]
        
            if parent is None:
                continue
            else:
                parent_indices = xsens_indices({'orientation' : [parent]})['orientation'][0]
                
                x_B = torch.tensor(self.segment_positions_in_parent_frame[segment], dtype=torch.float64)

                x_B = x_B.view(1, -1, 1).repeat(orientations.shape[0], 1, 1)

                R_GB = quat_to_rotMat(orientations[:,parent_indices])
                positions[i] = positions[self.index_of[parent]] + R_GB.bmm(x_B).squeeze(2)           
        
        return positions.permute(1, 0, 2) 
   
    def plot_single_position(self, position):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(projection='3d')
       
        colors = ['r', 'b', 'g', 'c', 'm', 'k']
        for i, indices in enumerate(self.skeleton_tree):
            xs = list(position[indices, 0])
            ys = list(position[indices, 1])
            zs = list(position[indices, 2])
            
            ax.scatter(xs, ys, zs, c=colors[i])
            ax.plot(xs, ys, zs, color=colors[i])

        ax.view_init(elev=0)
        ax.grid(False)
       
        # Setting the axes properties
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-1.0, 1.0])
        ax.set_zlabel('Z')

        plt.show()

    def animate_motion(self, orientations, azim, elev, title=None, continuous=False):
        if len(orientations.shape) == 1:
            orientations = orientations.unsqueeze(0)       
 
        def update_lines(num, data, lines):
            positions = data[num]
                
            for i, line in enumerate(lines):
                xs = list(positions[self.skeleton_tree[i], 0])
                ys = list(positions[self.skeleton_tree[i], 1])
                zs = list(positions[self.skeleton_tree[i], 2])
                
                line.set_data(xs, ys)
                line.set_3d_properties(zs)
                if continuous and (num % orientations.shape[0]) > orientations.shape[0]//2:
                    line.set_linestyle('--')
                else:
                    line.set_linestyle('-')
            return lines

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        data = self.forward_kinematics(orientations)

        lines = [ax.plot([0], [0], [0])[0] for _ in range(6)]
        limits = [-1.0, 1.0]
        
        self._setup_axis(ax, limits, title, azim, elev)

        line_ani = animation.FuncAnimation(fig, update_lines, frames=range(data.shape[0]), fargs=(data, lines),
										   interval=25, blit=True)
        plt.show()
        return line_ani

    def compare_motion(self, orientations, azim, elev, figname=None, titles=None):
        if len(orientations.shape) == 1:
            orientations = orientations.unsqueeze(0)

        def update_lines(num, data, lines):
            positions = data[num]

            for i, line in enumerate(lines):
                xs = list(positions[self.skeleton_tree[i], 0])
                ys = list(positions[self.skeleton_tree[i], 1])
                zs = list(positions[self.skeleton_tree[i], 2])

                line.set_data(xs, ys)
                line.set_3d_properties(zs)
            return lines

        num_plots = orientations.shape[0]
        fig = plt.figure(figsize=(orientations.shape[0]*3, 3))
        data = self.forward_kinematics(orientations)
        
        limits = [-1.0, 1.0]
        for i in range(orientations.shape[0]):
            ax = fig.add_subplot(1, orientations.shape[0], i+1, projection='3d')

            lines = [ax.plot([0], [0], [0])[0] for _ in range(6)]            

            self._setup_axis(ax, limits, azim, elev)

            if titles is not None:
                ax.set_title(titles[i])

            update_lines(i, data, lines)

        plt.subplots_adjust(wspace=0)
        plt.show()
        if figname:
            fig.savefig(figname, bbox_inches='tight')
        return fig

    def _setup_axis(self, ax, limits, azim, elev):
        ax.set_xlim3d(limits)
        ax.set_ylim3d(limits)
        ax.set_zlim3d(limits)

        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
            
        ax.view_init(azim=azim, elev=elev)
