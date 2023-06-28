#!/usr/bin/env python
# coding: utf-8

# # Comparison of simple and complex head models

# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
# import neuron
# import LFPy
from lfpykit.eegmegcalc import FourSphereVolumeConductor
from lfpykit import CellGeometry, CurrentDipoleMoment
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
# from ECSbook_simcode.neural_simulations import return_equidistal_xyz
# from ECSbook_simcode.plotting_convention import mark_subplots, simplify_axes
import h5py

head_colors = ["#ffb380", "#74abff", "#b3b3b3", "#c87137"]
radii_4s = [89000., 90000., 95000., 100000.]  # (µm)
sigmas_4s = [0.276, 1.65, 0.01, 0.465]  # (S/m)


# In[2]:


#def return_2d_eeg(eeg, x_2D, y_2D, radius):
#
#    N = 300
#    xi = np.linspace(np.min(x_2D), np.max(x_2D), N)
#    yi = np.linspace(np.min(y_2D), np.max(y_2D), N)
#    zi = scipy.interpolate.griddata((x_2D, y_2D), eeg,
#                                    (xi[None,:], yi[:,None]), method='linear')
#
#    dr = xi[1] - xi[0]
#    for i in range(N):
#        for j in range(N):
#            r = np.sqrt((xi[i])**2 + (yi[j])**2)
#            if (r - dr/2) > radius:
#                zi[j,i] = "nan"
#    return xi, yi, zi

#def plot_head_outline(ax, radius):
#    circle_npts = 100
#    head_x = radius * np.cos(np.linspace(0, 2 * np.pi, circle_npts))
#    head_y = radius * np.sin(np.linspace(0, 2 * np.pi, circle_npts))
#    patches = []
#    right_ear = mpatches.FancyBboxPatch([radius + 5000, -15000], 3000, 30000,
#        boxstyle=mpatches.BoxStyle("Round", pad=5000))
#    patches.append(right_ear)
#    left_ear = mpatches.FancyBboxPatch([-radius - 8000, -15000], 3000, 30000,
#        boxstyle=mpatches.BoxStyle("Round", pad=5000))
#    patches.append(left_ear)

#    collection = PatchCollection(patches, facecolor='none', edgecolor='k', alpha=1.0)
#    ax.add_collection(collection)
#    ax.plot(head_x, head_y, 'k')
#    ax.plot([radius])

#    ax.plot([-10000, 0, 10000], [radius, radius + 10000, radius], 'k')


#def plot_simple_head_model(ax, radius):

#    circle_npts = 100
#    head_x = radius * np.cos(np.linspace(0, 2 * np.pi, circle_npts))
#    head_y = radius * np.sin(np.linspace(0, 2 * np.pi, circle_npts))
#    patches = []

#    right_ear = mpatches.FancyBboxPatch([radius + radius / 20, -radius/10],
#                                        radius/50, radius/5,
#        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))
#    patches.append(right_ear)

#    left_ear = mpatches.FancyBboxPatch([-radius - radius / 20 - radius / 50, -radius / 10],
#                                       radius / 50, radius / 5,
#        boxstyle=mpatches.BoxStyle("Round", pad=radius/20))
#    patches.append(left_ear)

#    collection = PatchCollection(patches, facecolor='none',
#                                 edgecolor='k', alpha=1.0, lw=2)
#    ax.add_collection(collection)
#    ax.plot(head_x, head_y, 'k')
#    ax.plot([radius])

    # plot nose
#    ax.plot([-radius / 10, 0, radius  / 10], [radius, radius + radius/10, radius], 'k')


#def plot_cortical_crossection(nyhead, dipole_location):

    #nyhead.set_dipole_pos(dipole_location)
    #cortex_normal = nyhead.cortex_normal_vec
    # Plotting the results

    #x_lim = [-80, 80]
    #y_lim = [-140, 110]
    #z_lim = [-60, 110]

    #xticks = np.arange(x_lim[0], x_lim[-1] + 10, 10)
    #yticks = np.arange(y_lim[0], y_lim[-1] + 10, 10)
    #zticks = np.arange(z_lim[0], z_lim[-1] + 10, 10)

    #plt.close("all")
    #fig = plt.figure(figsize=[19, 10])
    #fig.suptitle("Dipole location: {}\nCortex normal: {}".format(
    #    nyhead.dipole_pos, cortex_normal))
    #fig.subplots_adjust(top=0.96, bottom=0.07, hspace=0.4,
    #                    wspace=0.2, left=0.05, right=0.99)
    #ax1 = fig.add_subplot(131, aspect=1, xlabel="x (mm)",
    #                      xticks=xticks, yticks=yticks,
    #                      ylabel='y (mm)', xlim=x_lim, ylim=y_lim)
    #ax2 = fig.add_subplot(132, aspect=1, xlabel="x (mm)", xticks=xticks,
    #                      yticks=zticks,
    #                      ylabel='z (mm)', xlim=x_lim, ylim=z_lim)
    #ax3 = fig.add_subplot(133, aspect=1, xlabel="y (mm)",
    #                      xticks=yticks, yticks=zticks,
    #                      ylabel='z (mm)', xlim=y_lim, ylim=z_lim)
    #[ax.grid(True) for ax in [ax1, ax2, ax3]]

    # Making cortical crossections
#    threshold = 2
#    xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] -
#                                    nyhead.dipole_pos[1]) < threshold)[0]
    #xy_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] -
    #                                nyhead.dipole_pos[2]) < threshold)[0]
    #yz_plane_idxs = np.where(np.abs(nyhead.cortex[0, :] -
    #                                nyhead.dipole_pos[0]) < threshold)[0]

    #ax1.scatter(nyhead.cortex[0, xy_plane_idxs], nyhead.cortex[1, xy_plane_idxs], s=5, c='k')
    #ax2.scatter(nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs], s=5, c='k')
    #ax3.scatter(nyhead.cortex[1, yz_plane_idxs], nyhead.cortex[2, yz_plane_idxs], s=5, c='k')

    #dipole_arrow = cortex_normal * 10
    #arrow_plot_params = dict(color = 'r',
    #                         lw=2,
    #                         head_width = 3)
    #print(cortex_normal)
    # Plotting dipole location
    #ax1.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], 'o', ms=5, color='r')
    #ax2.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], 'o', ms=5, color='r')
    #ax3.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], 'o', ms=5, color='r')

    #ax1.arrow(nyhead.dipole_pos[0], nyhead.dipole_pos[1],
    #          dipole_arrow[0], dipole_arrow[1], **arrow_plot_params)
    #ax2.arrow(nyhead.dipole_pos[0], nyhead.dipole_pos[2],
    #          dipole_arrow[0], dipole_arrow[2], **arrow_plot_params)
    #ax3.arrow(nyhead.dipole_pos[1], nyhead.dipole_pos[2],
    #          dipole_arrow[1], dipole_arrow[2], **arrow_plot_params)
#    return nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs]

#def plot_four_sphere_model(ax, radii, dipole_loc=None):
#
#    for i in range(4):
#        ax.add_patch(plt.Circle((0, 0), radius=radii[-1 - i],
#                                   color=head_colors[-1-i],
#                                   fill=True, ec='k', lw=.1))
#    if not dipole_loc is None:
#        ax.plot(dipole_loc[0], dipole_loc[2], 'r.')
#
#    # mark 4-sphere head model layers
#    ax.text(-10500, radii[0] - 3500, 'brain', ha="right", rotation=4)
#    ax.text(-10500, radii[1] - 2200, 'CSF', ha="right", rotation=4)
#    ax.text(-10500, radii[2] - 4000, 'skull', ha="right", rotation=4)
#    ax.text(-10500, radii[3] - 4000, 'scalp', ha="right", rotation=4)


# In[3]:


cdm_mag = 1e7
p = np.array([0, 0, cdm_mag])

dipole_locations = [
    # np.array([40, -38, 65.]),
    #np.array([40, -34, 70.]),
    #'parietal_lobe',
    #'calcarine_sulcus',
    #'occipital_lobe',
    # 'motorsensory_cortex'
    [37.8, -18.8, 71.1],
    [42.4, -18.8, 55.0],
]

# Prepare New York head model
nyhead = NYHeadModel()
elecs_NY = np.array(nyhead.elecs)
upper_idxs_NY = np.where(elecs_NY[2, :] > 0)[0]
elecs_x_NY = elecs_NY[0, upper_idxs_NY]
elecs_y_NY = elecs_NY[1, upper_idxs_NY]
elecs_z_NY = elecs_NY[2, upper_idxs_NY]
elecs_2D_NY = nyhead.head_data["locs_2D"]
num_elecs_NY = len(elecs_x_NY)

# Prepare foursphere head model

elecs_x_4s, elecs_y_4s, elecs_z_4s = return_equidistal_xyz(1000, radii_4s[-1] - 1)
upper_idxs_4s = np.where(elecs_z_4s > 0)
elecs_x_4s = elecs_x_4s[upper_idxs_4s]
elecs_y_4s = elecs_y_4s[upper_idxs_4s]
elecs_z_4s = elecs_z_4s[upper_idxs_4s]
num_elecs_4s = len(elecs_x_4s)
r_elecs_4s = np.vstack((elecs_x_4s, elecs_y_4s, elecs_z_4s)).T
sphere_model = FourSphereVolumeConductor(r_elecs_4s, radii_4s, sigmas_4s)

eegs_NY = []
# eegs_4s = []
dipole_locs = []
dipole_vecs = []

for i, dipole_location in enumerate(dipole_locations):

    nyhead.set_dipole_pos(dipole_location)
    # We rotate current dipole moment to be oriented along the normal vector of cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    dipole_locs.append(nyhead.dipole_pos)
    dipole_vecs.append(p)
    print(nyhead.dipole_pos)

    M_NY = nyhead.get_transformation_matrix()
    eeg_NY = M_NY @ p * 1e3 # [mV -> µV]
    eegs_NY.append(eeg_NY[upper_idxs_NY])

    M_4s = sphere_model.get_transformation_matrix(np.array(nyhead.dipole_pos) * 1e3)
    eeg_4s = 1e3 * M_4s @ p # (uV)
    eegs_4s.append(eeg_4s)


# In[4]:


x_lim = [-105, 105]
y_lim = [-120, 110]
z_lim = [-100, 110]

plt.close("all")
fig = plt.figure(figsize=[6, 4])
fig.subplots_adjust(right=0.89, bottom=0.04, left=0.03, wspace=0.1, hspace=0.1, top=.98)

ax_dict = dict(frameon=False, xticks=[], yticks=[], aspect=1)

ax0_NY = fig.add_subplot(231, xlabel="x (mm)", ylabel='z (mm)',
                         xlim=x_lim, ylim=z_lim, **ax_dict)
ax1_NY = fig.add_subplot(232, xlabel="x (mm)", ylabel='y (mm)',
                  xlim=x_lim, ylim=y_lim, **ax_dict)
ax2_NY = fig.add_subplot(233, xlabel="x (mm)", ylabel='y (mm)',
                         xlim=x_lim, ylim=y_lim,**ax_dict)

ax0_4s = fig.add_subplot(234, xlabel="x (mm)", ylabel='z (mm)',
                         xlim=x_lim, ylim=z_lim, **ax_dict)
ax1_4s = fig.add_subplot(235, xlabel="x (mm)", ylabel='y (mm)',
                  xlim=x_lim, ylim=y_lim, **ax_dict)
ax2_4s = fig.add_subplot(236, xlabel="x (mm)", ylabel='y (mm)',
                         xlim=x_lim, ylim=y_lim, **ax_dict)

threshold = 1
xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] -
                            nyhead.dipole_pos[1]) < threshold)[0]
cortex_x, cortex_z = nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs]
ax0_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])

arrow_plot_params = dict(lw=2, head_width=3, zorder=1000)
for i, dipole_location in enumerate(dipole_locations):
    color = 'rb'[i]
    dipole_arrow = dipole_vecs[i] / np.linalg.norm(dipole_vecs[i]) * 10
    ax0_NY.arrow(dipole_location[0], dipole_location[2],
                  dipole_arrow[0], dipole_arrow[2], color=color, **arrow_plot_params)
    ax0_4s.arrow(dipole_location[0], dipole_location[2],
                  dipole_arrow[0], dipole_arrow[2], color=color, **arrow_plot_params)

    ax_4s = [ax1_4s, ax2_4s][i]
    ax_NY = [ax1_NY, ax2_NY][i]
    ax_4s.plot(dipole_location[0], dipole_location[1], 'o', color=color, zorder=10)
    ax_NY.plot(dipole_location[0], dipole_location[1], 'o', color=color, zorder=10)

head = np.array(nyhead.head_data["head"]["vc"])
threshold = 10
xz_plane_idxs = np.where(np.abs(head[1, :] -
                            nyhead.dipole_pos[1]) < threshold)[0]
head_x, head_z = head[0, xz_plane_idxs], head[2, xz_plane_idxs]

ax0_NY.scatter(head_x, head_z, s=4, c=head_colors[-1])

threshold = 10
xz_plane_idxs = np.where(np.abs(elecs_y_NY -
                            nyhead.dipole_pos[1]) < threshold)[0]
eeg_x, eeg_z = head[0, xz_plane_idxs], head[2, xz_plane_idxs]

ax0_NY.scatter(elecs_x_NY[xz_plane_idxs], elecs_z_NY[xz_plane_idxs], s=15)


for i in range(4):
    ax0_4s.add_patch(plt.Circle((0, 0), radius=radii_4s[-1 - i] / 1000,
                               color=head_colors[-1-i],
                               fill=True, ec='k', lw=.1))

print("Max four-sphere, loc #1: {:1.3f}".format(np.max(np.abs(eegs_4s[0]))))
print("Max New York, loc #1: {:1.3f}".format(np.max(np.abs(eegs_NY[0]))))

print("Max four-sphere, loc #2: {:1.3f}".format(np.max(np.abs(eegs_4s[1]))))
print("Max New York, loc #2: {:1.3f}".format(np.max(np.abs(eegs_NY[1]))))


vmax = 1.0#np.max(np.abs(eeg))

cmap = plt.cm.get_cmap('PRGn')
vmap = lambda v: cmap((v + vmax) / (2*vmax))
levels = np.linspace(-vmax, vmax, 60)

contourf_kwargs = dict(levels=levels,
                       cmap="PRGn",
                       vmax=vmax,
                       vmin=-vmax)

contour_kwargs = dict(levels=levels,
                       cmap="PRGn",
                       vmax=vmax,
                       vmin=-vmax,
                     linewidths=1)

img_NY = ax2_NY.tricontourf(elecs_x_NY, elecs_y_NY,
                            eegs_NY[1] / np.max(np.abs(eegs_NY[1])),
                            **contourf_kwargs)
ax2_NY.tricontour(elecs_x_NY, elecs_y_NY,
                            eegs_NY[1] / np.max(np.abs(eegs_NY[1])),
                            **contour_kwargs)
img_4s = ax2_4s.tricontourf(elecs_x_4s / 1000, elecs_y_4s / 1000,
                            eegs_4s[1] / np.max(np.abs(eegs_4s[1])),
                            **contourf_kwargs)
ax2_4s.tricontour(elecs_x_4s / 1000, elecs_y_4s / 1000,
                            eegs_4s[1] / np.max(np.abs(eegs_4s[1])),
                            **contour_kwargs)
img_NY = ax1_NY.tricontourf(elecs_x_NY, elecs_y_NY,
                            eegs_NY[0] / np.max(np.abs(eegs_NY[0])),
                            **contourf_kwargs)
ax1_NY.tricontour(elecs_x_NY, elecs_y_NY,
                            eegs_NY[0] / np.max(np.abs(eegs_NY[0])),
                            **contour_kwargs)

img_4s = ax1_4s.tricontourf(elecs_x_4s / 1000, elecs_y_4s / 1000,
                            eegs_4s[0] / np.max(np.abs(eegs_4s[0])),
                            **contourf_kwargs)
ax1_4s.tricontour(elecs_x_4s / 1000, elecs_y_4s / 1000,
                            eegs_4s[0] / np.max(np.abs(eegs_4s[0])),
                            **contour_kwargs)

#plot_simple_head_model(ax2, radius * 0.7)
mark_subplots(fig.axes, ypos=1.0, xpos=0.)
cax = fig.add_axes([0.89, 0.2, 0.01, 0.6])
cbar = plt.colorbar(img_NY, cax=cax, label="nornmalized",)
cbar.set_ticks(np.linspace(-int(vmax), int(vmax), 9))

plt.savefig("comparison_simple_and_complex_head_models.pdf")






