import matplotlib.pyplot as plt
import numpy as np
import os

# labels = ['Lime-PurpleBlue', 'LimePurple-Green', 'Nothing-Blue', 'Orange-RedSilver', 'PurpleRed-Red', 'WhiteSilver-Pink', 'Yellow-GreenPurple', 'YellowPurple-Yellow']

# for i in labels:
#     imgs = os.listdir(f"./heatmap/{i}")
#     for j in imgs:
#         heatmap = np.load(f"./heatmap/{i}/{j}", allow_pickle=True)

#         if j[:3] == "neg":
#             plt.imshow(heatmap, cmap='Reds_r')
#         elif j[:3] == "pos":
#             plt.imshow(heatmap, cmap='Greens')
#         elif j[:3] == "abs":
#             plt.imshow(heatmap, cmap='Greys')
#         else:
#             plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
#         plt.colorbar()
#         plt.savefig(f"./heatmap_imgs/{i}/{j[:-4]}.png")
#         plt.cla()
#         plt.clf()

basepath = "./explanations2.0/heatmap"

for i in os.listdir(basepath):
    print(i)
    path = os.path.join(basepath, i)
    if os.path.isdir(path):
        imgs = os.listdir(f"./explanations2.0/heatmap/{i}")
        for j in imgs:
            heatmap = np.load(f"./explanations2.0/heatmap/{i}/{j}", allow_pickle=True)

            if j[:3] == "neg":
                plt.imshow(heatmap, cmap='Reds_r')
            elif j[:3] == "pos":
                plt.imshow(heatmap, cmap='Greens')
            elif j[:3] == "abs":
                plt.imshow(heatmap, cmap='Greys')
            else:
                plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
            plt.colorbar()
            plt.savefig(f"./explanations2.0/heatmap_imgs/{i}/{j[:-4]}.png")
            plt.cla()
            plt.clf()
    else:
        heatmap = np.load(f"./explanations2.0/heatmap/{i}", allow_pickle=True)

        if i[:3] == "neg":
            plt.imshow(heatmap, cmap='Reds_r')
        elif i[:3] == "pos":
            plt.imshow(heatmap, cmap='Greens')
        elif i[:3] == "abs":
            plt.imshow(heatmap, cmap='Greys')
        else:
            plt.imshow(heatmap, cmap="RdBu", vmin=-heatmap.max(), vmax=heatmap.max())
        plt.colorbar()
        plt.savefig(f"./explanations2.0/heatmap_imgs/{i[:-4]}.png")
        plt.cla()
        plt.clf()