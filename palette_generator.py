import seaborn as sns
import extcolors
import numpy as np
from colorsys import hls_to_rgb, rgb_to_hls
from math import ceil

class PaletteExtractor:
    def __init__(self, img_path):
        self.colors, _ = extcolors.extract_from_path(img_path)
        self.palette = None

    def extract_colors(self, color_count=None, exclude_colors=[(255, 255, 255), (0, 0, 0)], exact_exclusion=True, t=5):
        colors = [i[0] for i in self.colors]
        if exclude_colors and exact_exclusion:
            set_colors = set(colors)
            # exclude precise colors
            for clr in exclude_colors:
                if clr not in colors:
                    continue
                colors.remove(clr)

            self.colors = np.array(list(set_colors))
        elif exclude_colors and not exact_exclusion:
            colors = np.array(colors)

        
            logicals = [np.logical_and(colors <= i+t, colors >= i-t) for i in np.array(exclude_colors)]
            logical_exclusion = np.logical_or.reduce(logicals)

            mask = ~np.all(logical_exclusion, axis=1)
            self.colors = colors[mask]
    
    def sort_palette(self):
        # define palette and sort by its Lightness parameter, colors is in (r,g,b) format
        palette_sorted_hls = [rgb_to_hls(r, g, b) for r, g, b in self.colors]
        palette_sorted_hls.sort(key=lambda x: x[1])
        palette_sorted_rgb = [hls_to_rgb(h, l, s) for h, l, s in palette_sorted_hls]
        palette_sorted_rgb = [tuple(map(ceil, rgbs_)) for rgbs_ in palette_sorted_rgb]

        self.palette = palette_sorted_rgb

    def interpolate_palette(self, total=100):
        interpolated_palette = []
        palette = self.palette

        # mean absolute differences between adjacent colors in the initial palette across each channel (r,g,b)
        # we need that to estimate the amount of additional colors to insert to achieve max overall smoothness
        abs_spacings = []
        for i in range(len(palette)-1):
            left_clr, right_clr = np.array(palette[i]), np.array(palette[i+1])
            
            mae = abs(np.mean(left_clr - right_clr))
            abs_spacings.append(mae)
        
        n_colors_to_spacing = []
        total_mae = sum(abs_spacings)
        for mae in abs_spacings:
            # between each pair of adjacent colors in palette, add the amount of colors to smooth proportional to absolute difference between these adjacent colors
            n_colors_to_spacing.append((mae/total_mae)*(total-len(palette)))

        # transform into integers
        n_colors_to_spacing = [int(i) for i in n_colors_to_spacing]
        if total-len(palette) > sum(n_colors_to_spacing):
            
            n_lacking = total - len(palette) - sum(n_colors_to_spacing)
            for i in range(n_lacking):
                n_colors_to_spacing[i] += 1
        

        # insert colors in each color channel
        for i in range(len(n_colors_to_spacing)):
            # from initial adjacent colors, append the left one
            adjacent = (palette[i], palette[i+1])
            interpolated_palette.append(adjacent[0])

            # insert colors in between
            insert_per_channel = lambda left_, right_, step_: np.linspace(left_, right_, step_+2)[1:-1]
            colors_all_channels = [insert_per_channel(adjacent[0][channel], adjacent[1][channel], n_colors_to_spacing[i]) for channel in range(3)]
            # into 3d array
            colors_all_channels = np.stack(colors_all_channels, axis=1)
            interpolated_palette.extend(colors_all_channels)
        # insert the rightmost element
        interpolated_palette.append(palette[-1])
        self.palette = sns.color_palette(np.array(interpolated_palette)/255)

    def smooth_palette(self, color_count=None, exclude_colors=[(255, 255, 255), (0, 0, 0)], exact_exclusion=True, t=5, total=100):
        self.extract_colors(color_count=color_count, exclude_colors=exclude_colors, exact_exclusion=exact_exclusion, t=t)
        self.sort_palette()
        self.interpolate_palette(total=total)

    def draw_equally(self, k=3):
        palette = self.palette
        # two colors are most and least bright
        n_splits = (k-2) + 1
        palette_len = len(palette)
        clrs_per_split = int(palette_len/n_splits)
        clrs_indices = [i*clrs_per_split for i in range(1, k-1)]
        colors_drawn = np.array(palette)[clrs_indices]
        colors_drawn = np.concatenate([[palette[0]], colors_drawn, [palette[-1]]])
        return sns.color_palette(colors_drawn)
        