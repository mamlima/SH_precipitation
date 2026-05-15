#%% 0. Start
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cf
from io import BytesIO
from PIL import Image
import pickle

which_season = 'year'

#%% 1. Functions


#%% 2. Open data
#%%% 2.1. Open composite data
z500 = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_Z500_{which_season}.nc')
mslp = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_MSLP_{which_season}.nc')
ivt = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_IVT_{which_season}.nc')
iwv = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_IWV_{which_season}.nc')
UV = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_UV_{which_season}.nc')
OLR = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_OLR_{which_season}.nc')

with open(f'../Data/Composites/Correlation_Z500_{which_season}.pickle', 'rb') as handle:
    z500_corr = pickle.load(handle)
with open(f'../Data/Composites/Correlation_MSLP_{which_season}.pickle', 'rb') as handle:
    mslp_corr = pickle.load(handle)
with open(f'../Data/Composites/Correlation_IVT_{which_season}.pickle', 'rb') as handle:
    ivt_corr = pickle.load(handle)
with open(f'../Data/Composites/Correlation_IWV_{which_season}.pickle', 'rb') as handle:
    iwv_corr = pickle.load(handle)
with open(f'../Data/Composites/Correlation_UV_{which_season}.pickle', 'rb') as handle:
    UV_corr = pickle.load(handle)
with open(f'../Data/Composites/Correlation_OLR_{which_season}.pickle', 'rb') as handle:
    OLR_corr = pickle.load(handle)

#%%% 2.2. Create lat, lon array
lat = z500.lat.values
lon = z500.lon.values

#%%% 2.3. Choose interest boxes
#### For subtropical ridges
latmin_r, latmax_r = -55, -30
step_r = 30; lons_r = np.arange(-180,180+step_r,step_r)
lonbounds_r =  {}
for i in range(len(lons_r)-1):
    lonbounds_r[f'lonmin_{i+1}'] = lons_r[i]
    lonbounds_r[f'lonmax_{i+1}'] = lons_r[i+1]


#%% 3. Figures
#%%% 3.1. Function to draw axes (returns ax and mappable; colorbar placed externally)
def indv_ax(gs_slot,
            c_anom, c_sign, c_abs, 
            vmin, vmax, step, var_lvls, lbl_step, cmap,
            add_xax, add_yax, lbl):
    
    ax = fig.add_subplot(gs_slot)
    ax.grid(ls='--', alpha=0.4, color='grey')
    
    #### Colors (anomaly)
    im = ax.contourf(lon, lat,  c_anom,
                      levels = np.arange(vmin, vmax+step, step),
                      cmap = cmap, alpha = 0.9, zorder=1, extend = 'both')
    
    #### Hatches (Significance)
    ax.contourf(lon, np.arange(-45,46,1)[::-1],  np.where(c_sign,1,0),
                hatches='oo', zorder=1, levels=[0.9,1.1], cmap='Greys', alpha=0)

    #### Contours (Absolut field)
    CS = ax.contour(lon, lat, c_abs, 
                    alpha=0.65, levels=var_lvls, colors = 'black',
                    linewidths=np.where(var_lvls%lbl_step == 0, 1., 0.1))
    label_levels = [level for level in CS.levels if level % lbl_step == 0]
    labels = ax.clabel(CS, label_levels, inline=True, fontsize=8)
    for label in labels:
        label.set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'white', 'pad': 1})
    
    # Create a rectangle patch for west
    latmin_r, latmax_r = -55, -30
    rect = patches.Rectangle((-15, -(latmax_r - latmin_r)/2),
                             30, latmax_r - latmin_r,
                             linewidth=1.8, ls='-', edgecolor='black', facecolor='none', zorder=100)
    ax.add_patch(rect)

    # Add limits and labels to axes
    ax.set_ylim(-45,45)
    if add_xax:
        ax.set_xlabel('Lon [$^{\circ}$] relative to centre', fontsize=14)
        ax.set_xticks(np.arange(-180,180+45,45))
    else:
        ax.set_xticks(np.arange(-180,180+45,45),['' for i in np.arange(-180,180+45,45)])
    
    if add_yax:
        ax.set_ylabel('Lat [$^{\circ}$] relative to centre', fontsize=14)
        ax.set_yticks(np.arange(-40,40+20,20))
    else:
        ax.set_yticks(np.arange(-40,40+20,20),['' for i in np.arange(-40,40+20,20)])
    
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    #### Letter
    ax.annotate(lbl,  xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize=18, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', pad=3.0),
                zorder=200)

    return ax, im


#%%% 3.1b. Legacy function for supplementary figure (keeps old interface)
def indv_ax_legacy(lines, cols, pos,
            c_anom, c_sign, c_abs, 
            vmin, vmax, step, var_lvls, lbl_step, cmap,
            cbar_ticks, cbar_label,
            add_xax, add_yax, colorbar_loc, lbl,
            cbar_orientation='horizontal'):
    
    ax = fig.add_subplot(lines,cols,pos)
    ax.grid(ls='--', alpha=0.4, color='grey')
    
    im = ax.contourf(lon, lat,  c_anom,
                      levels = np.arange(vmin, vmax+step, step),
                      cmap = cmap, alpha = 0.9, zorder=1, extend = 'both')
    ax.contourf(lon, np.arange(-45,46,1)[::-1],  np.where(c_sign,1,0),
                hatches='oo', zorder=1, levels=[0.9,1.1], cmap='Greys', alpha=0)
    CS = ax.contour(lon, lat, c_abs, 
                    alpha=0.65, levels=var_lvls, colors = 'black',
                    linewidths=np.where(var_lvls%lbl_step == 0, 1., 0.1))
    label_levels = [level for level in CS.levels if level % lbl_step == 0]
    labels = ax.clabel(CS, label_levels, inline=True, fontsize=8)
    for label in labels:
        label.set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'white', 'pad': 1})
    
    latmin_r, latmax_r = -55, -30
    rect = patches.Rectangle((-15, -(latmax_r - latmin_r)/2),
                             30, latmax_r - latmin_r,
                             linewidth=1.8, ls='-', edgecolor='black', facecolor='none', zorder=100)
    ax.add_patch(rect)

    ax.set_ylim(-45,45)
    if add_xax:
        ax.set_xlabel('Lon [$^{\circ}$] relative to centre', fontsize=14)
        ax.set_xticks(np.arange(-180,180+45,45))
    else:
        ax.set_xticks(np.arange(-180,180+45,45),['' for i in np.arange(-180,180+45,45)])
    if add_yax:
        ax.set_ylabel('Lat [$^{\circ}$] relative to centre', fontsize=14)
        ax.set_yticks(np.arange(-40,40+20,20))
    else:
        ax.set_yticks(np.arange(-40,40+20,20),['' for i in np.arange(-40,40+20,20)])
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    cax = fig.add_axes(colorbar_loc)
    cbar = fig.colorbar(im, cax = cax, ticks = cbar_ticks,
                        orientation=cbar_orientation, extend = 'both', pad = 0.45, aspect = 25)
    cbar.set_label(cbar_label, fontsize=15, labelpad=8)
    cbar.ax.tick_params(labelsize=14)
    
    ax.annotate(lbl,  xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize=18, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', pad=3.0),
                zorder=200)

    return ax

#%%% 3.2. Helper function for violin plots
def draw_violin(box_ax, data, position_offset, label_name, c):
    alpha = 0.6
    lw = .8
    # Use position_offset to space the 4 variables within each sector
    parts = box_ax.violinplot(data, positions=np.arange(-180+30/8*position_offset, 180 + (7.5 if position_offset == 7 else 0), 30),
                              showmedians=True, widths=6)
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_facecolor(c)
        pc.set_alpha(alpha)
    color = parts['bodies'][0].get_facecolor().flatten()
    patch_label = (patches.Patch(color=color), label_name)
    # Vertical bar (min–max)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(lw)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(lw)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(lw)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(lw)
    return patch_label

#%%% 3.3. Helper function for the SH map + violin bottom panel
def draw_map_and_violins(fig, data_list, var_names, panel_letter, gs_slot=None):
    n_vars = len(data_list)
    section_names = [f'R{w}' for w in range(1, 13)]
    
    # ---------------------------------------
    # STEP 1 — Generate SH map into an image
    # ---------------------------------------
    fig_map = plt.figure(figsize=(8, 4))
    ax_map = fig_map.add_subplot(111, projection=ccrs.PlateCarree())
    ax_map.set_extent([-180, 180, -85, 0], ccrs.PlateCarree())
    land = cf.NaturalEarthFeature(category='physical',
                                  scale='50m',
                                  facecolor='gainsboro', name='land')
    ax_map.add_feature(land, edgecolor='k', linewidth=0.5, alpha=0.9, zorder=0)
    borders = cf.NaturalEarthFeature(category='cultural',
                                      name='admin_0_boundary_lines_land',
                                      scale='50m',
                                      facecolor='none', edgecolor='black')
    ax_map.add_feature(borders, edgecolor='k', linewidth=0.2, alpha=0.9, zorder=0)
    ax_map.gridlines(draw_labels=False, linestyle="--", alpha=0.3)
    
    ax_map.plot([-180,180], [latmax_r,latmax_r], linewidth=1., ls='-',
              color='dimgrey', transform=ccrs.PlateCarree(), zorder=1)
    ax_map.plot([-180,180], [latmin_r,latmin_r], linewidth=1., ls='-',
              color='dimgrey', transform=ccrs.PlateCarree(), zorder=1)
    for j,i in enumerate(lons_r[:-1]):
        ax_map.plot([i,i], [latmin_r,latmax_r], linewidth=1, ls='--',
                  color='dimgrey', transform=ccrs.PlateCarree(), zorder=1)    
    
    # Save map to image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close(fig_map)
    buf.seek(0)
    
    # Load as PIL image
    bg_img = np.array(Image.open(buf))
    
    # ---------------------------------------
    # STEP 2 — Create final figure with map 
    #          as background + boxplots
    # ---------------------------------------
    if gs_slot is not None:
        ax = fig.add_subplot(gs_slot)
    else:
        ax = fig.add_subplot(3, 2, (5, 6))
    
    # draw SH map behind
    ax.imshow(bg_img, aspect='auto', extent=[0, 1, 0, 1], zorder=0)
    ax.axis('off')
    
    # Create overlay axis for boxplots (same coordinates)
    box_ax = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes, zorder=2)
    
    labels = []
    # Dynamically space variables within each 30° sector
    all_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:cyan']
    if n_vars == 3:
        offsets = [1, 4, 7]
    elif n_vars == 4:
        offsets = [1, 3, 5, 7]
    else:
        offsets = list(range(1, 2*n_vars, 2))[:n_vars]
    
    for data, name, offset, which in zip(data_list, var_names, offsets, range(n_vars)):
        patch_label = draw_violin(box_ax, data, offset, name, all_colors[which])
        labels.append(patch_label)
    
    box_ax.set_xticks(range(-165, 180, 30))
    box_ax.set_xlim(-180, 180)
    box_ax.set_xticklabels(section_names, rotation=45, fontsize=14)
    box_ax.set_yticks(np.arange(0, 1.2, 0.2))
    box_ax.set_ylim(0, 1)
    box_ax.set_yticklabels(labels=[f'{i.round(1)}' for i in np.arange(0, 1.2, 0.2)], fontsize=14)
    box_ax.set_ylabel("Coef. of determination ($r^2$)", fontsize=14)
    
    # Make background transparent so map shows behind
    box_ax.patch.set_alpha(0)
    
    # Remove spines so map shows cleanly
    for spine in box_ax.spines.values():
        spine.set_visible(False)
    
    #### Letter
    ax.annotate(panel_letter,  xy=(0, 1.167), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize=18, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', pad=3.0), zorder=100)
    
    leg = plt.legend(*zip(*labels), ncol=n_vars, bbox_to_anchor=(0.04, 0.99),
               loc='lower left', fontsize=15)
    for lh in leg.legend_handles:
        lh.set_alpha(1)


##########################################################################################################################################
#%% 3.4. Draw Figure 1 — Main paper figure: Z500, OLR, IVT + Coef. of determination
##########################################################################################################################################
plt.close('all')
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 17))

# Outer GridSpec: 2 rows — top block (rows 1-3) and bottom (row 4)
gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                             height_ratios=[3, 1],
                             hspace=0.2)

# Inner GridSpec for the top 3 plots (single column — no colorbar column)
gs_top = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_outer[0],
                                          width_ratios=[4,0.2],
                                          hspace=0.065)

# Inner GridSpec for the bottom violin panel
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_outer[1])

# Custom IVT colormap
colors_ivt = ["#804434","#AF5235","#E5673E","#EAA549","#EEC651","#F9F86F","#EDF6A5","#EFF2D3","#FFFFFF",
              "#DDF2EE","#CCF3E9","#A9DBF4","#A1B4F2","#536FF4","#5034E5","#8D38D7","#693691"]
cmap_ivt = mcolors.LinearSegmentedColormap.from_list("ivt", colors_ivt)

# --- Row 1: Z500 ---
ax1, im1 = indv_ax(gs_top[0],
              z500.c_anom.values, z500.c_sign.values, z500.c_abs.values,
              -115, 115, 10, np.arange(300,7000,100), 100, 'RdGy_r',
              False, True, 'a)')

# --- Row 2: OLR ---
ax2, im2 = indv_ax(gs_top[2],
              OLR.c_anom.values, OLR.c_sign.values, OLR.c_abs.values,
              -8.5, 8.5, 1, np.arange(0,400,10), 20, 'PuOr_r',
              False, True, 'b)')

# --- Row 3: IVT ---
ax3, im3 = indv_ax(gs_top[4],
              ivt.c_anom.values, ivt.c_sign.values, ivt.c_abs.values,
              -55, 55, 5, np.arange(0,500,25), 50, cmap_ivt,
              True, True, 'c)')

# --- Manually place vertical colorbars aligned to each plot ---
# Format: [left, bottom, width, height] in figure coordinates
# Adjust these values to fine-tune colorbar position
cbar_width = 0.018
cbar_left  = 0.82   # <-- move left/right to fit inside bottom panel limits

# Force a layout pass so get_position() returns final coordinates
fig.canvas.draw()

for ax_i, im_i, ticks, label in [
    (ax1, im1, [-115,-75,-35,0,35,75,115],      'Z500 diff [m]'),
    (ax2, im2, [-8.5,-4.5,0,4.5,8.5],           'OLR diff [$W\ m^{-2}$]'),
    (ax3, im3, [-55,-25,0,25,55],                r'IVT diff [kg m$^{-1}$ s$^{-1}$]'),
]:
    pos = ax_i.get_position()
    cax = fig.add_axes([cbar_left, pos.y0, cbar_width, pos.height])
    cbar = fig.colorbar(im_i, cax=cax, ticks=ticks,
                        orientation='vertical', extend='both')
    cbar.set_label(label, fontsize=15, labelpad=10)
    cbar.ax.tick_params(labelsize=13)

# --- Row 4: Violin plots (Coef. of determination): Z500, OLR, IVT ---
sections = [f"section_{i}" for i in range(1, 13)]
data_z500 = [z500_corr[s] for s in sections]
data_olr = [OLR_corr[s] for s in sections]
data_ivt = [ivt_corr[s] for s in sections]

draw_map_and_violins(fig,
                     [data_z500, data_olr, data_ivt],
                     ['Z500', 'OLR', 'IVT'],
                     'd)',
                     gs_slot=gs_bot[0])

# Save
plt.savefig(f'../Figures/Fig4_Composite_(Main)_{which_season}.jpg', dpi=600, bbox_inches='tight')

#%% 3.4. Draw Figure 3 — Supp paper figure: MSLP, IWV, UV + Coef. of determination
##########################################################################################################################################
plt.close('all')
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 17))

# Outer GridSpec: 2 rows — top block (rows 1-3) and bottom (row 4)
gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                             height_ratios=[3, 1],
                             hspace=0.2)

# Inner GridSpec for the top 3 plots (single column — no colorbar column)
gs_top = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_outer[0],
                                          width_ratios=[4,0.2],
                                          hspace=0.065)

# Inner GridSpec for the bottom violin panel
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_outer[1])

# Custom IVT colormap
colors_ivt = ["#804434","#AF5235","#E5673E","#EAA549","#EEC651","#F9F86F","#EDF6A5","#EFF2D3","#FFFFFF",
              "#DDF2EE","#CCF3E9","#A9DBF4","#A1B4F2","#536FF4","#5034E5","#8D38D7","#693691"]
cmap_ivt = mcolors.LinearSegmentedColormap.from_list("ivt", colors_ivt)

# --- Row 1: Z500 ---
ax1, im1 = indv_ax(gs_top[0],
              mslp.c_anom.values, mslp.c_sign.values, mslp.c_abs.values,
              -10.5, 10.5, 1, np.arange(900,1040,10), 10, 'RdGy_r',
              False, True, 'a)')

# --- Row 2: OLR ---
ax2, im2 = indv_ax(gs_top[2],
              iwv.c_anom.values, iwv.c_sign.values, iwv.c_abs.values,
              -2.0, 2.0, 0.2, np.arange(0,50,2.5), 5, 'BrBG',
              False, True, 'b)')

# --- Row 3: IVT ---
ax3, im3 = indv_ax(gs_top[4],
              UV.c_anom.values, UV.c_sign.values, UV.c_abs.values,
              -2.5, 2.5, 0.2, np.arange(0,50,2.5), 2.5, 'bwr',
              True, True, 'c)')

# --- Manually place vertical colorbars aligned to each plot ---
# Format: [left, bottom, width, height] in figure coordinates
# Adjust these values to fine-tune colorbar position
cbar_width = 0.018
cbar_left  = 0.82   # <-- move left/right to fit inside bottom panel limits

# Force a layout pass so get_position() returns final coordinates
fig.canvas.draw()

for ax_i, im_i, ticks, label in [
    (ax1, im1, [-10.5,-6.5,-2.5,2.5,6.5,10.5],      'MSLP diff [hPa]'),
    (ax2, im2, [-2,-1.2,-0.4,0.4,1.2,2],           r'IWV diff [kg m$^{-2}$]'),
    (ax3, im3, [-2.5,-1.3,0,1.3,2.5],               'UV diff [$m\ s^{-1}$]'),
]:
    pos = ax_i.get_position()
    cax = fig.add_axes([cbar_left, pos.y0, cbar_width, pos.height])
    cbar = fig.colorbar(im_i, cax=cax, ticks=ticks,
                        orientation='vertical', extend='both')
    cbar.set_label(label, fontsize=15, labelpad=10)
    cbar.ax.tick_params(labelsize=13)

# --- Row 4: Violin plots (Coef. of determination): Z500, OLR, IVT ---
sections = [f"section_{i}" for i in range(1, 13)]
data_mslp = [mslp_corr[s] for s in sections]
data_iwv = [iwv_corr[s] for s in sections]
data_UV = [UV_corr[s] for s in sections]

draw_map_and_violins(fig,
                     [data_mslp, data_iwv, data_UV],
                     ['MSLP', 'IWV', 'UV'],
                     'd)',
                     gs_slot=gs_bot[0])

# Save
plt.savefig(f'../Figures/Fig4_Composite_(Supplementary)_{which_season}.jpg', dpi=600, bbox_inches='tight')