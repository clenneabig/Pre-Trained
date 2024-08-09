from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import timm
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

def batch_predict_gen(l):
    def batch_predict(images):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        files = glob.glob('./images_array/*')

        for f in files:
            os.remove(f)

        all_imgs = []
        for i in range(len(images)):
            path = f"/home/clenneabig/Desktop/AIML591/Pre-Trained/images_array/image{str(i)}.npy"
            all_imgs.append(path)
            np.save(path, images[i])

        df = pd.DataFrame({
            'image_id': list(range(1, len(all_imgs)+1)),
            'identity': [l for i in range(len(all_imgs))],
            'path': all_imgs,
        })

        transform = T.Compose([T.Resize(size=(384, 384)), 
                                    T.ToTensor(), 
                                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

        df_images = WildlifeDataset(df, transform=transform, img_load='crop_black')

        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)

        extractor = DeepFeatures(model, device=device, num_workers=2)

        database = np.load("./database.npy", allow_pickle=True)
        query = extractor(df_images)

        sim_func = CosineSimilarity()
        sim = sim_func(query, database)['cosine']

        labels_string = np.load("./database_labels.npy", allow_pickle=True)

        clas_func = KnnClassifier(k=5, database_labels=labels_string)
        pred, probs = clas_func(sim)

        return np.array(probs)
    return batch_predict




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if torch.cuda.is_available():
        torch.set_default_device(device)
        torch.cuda.empty_cache()


    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
            
    paths = []

    with open("./query_paths.txt", 'r') as fp:
        for line in fp:
            splits = line.split('/')
            paths.append((splits[7], splits[8][:-1]))
    
    label_to_idx = {
        "Lime-PurpleBlue" : 0,
        "LimePurple-Green" : 1,
        "Nothing-Blue" : 2,
        "Orange-RedSilver" : 3,
        "PurpleRed-Red" : 4,
        "WhiteSilver-Pink" : 5,
        "Yellow-GreenPurple" : 6,
        "YellowPurple-Yellow" : 7
    }

    idx_to_label = {
        0 : "Lime-PurpleBlue",
        1 : "LimePurple-Green",
        2 : "Nothing-Blue",
        3 : "Orange-RedSilver",
        4 : "PurpleRed-Red",
        5 : "WhiteSilver-Pink",
        6 : "Yellow-GreenPurple",
        7 : "YellowPurple-Yellow"
    }

    size = 384

    total_abs_avg = np.zeros(shape=(size, size)).tolist()
    total_pos_avg = np.zeros(shape=(size, size)).tolist()
    total_neg_avg = np.zeros(shape=(size, size)).tolist()

    abs_avg = np.zeros(shape=(size, size)).tolist()
    pos_avg = np.zeros(shape=(size, size)).tolist()
    neg_avg = np.zeros(shape=(size, size)).tolist()

    label_avgs = [[], [], [], [], [], [], [], []]

    current_label = ""

    for label, img in paths:
        if current_label == "":
            current_label = label
        elif current_label != label:
            label_avgs[label_to_idx[current_label]].append(abs_avg)
            label_avgs[label_to_idx[current_label]].append(pos_avg)
            label_avgs[label_to_idx[current_label]].append(neg_avg)
            abs_avg = np.zeros(shape=(size, size)).tolist()
            pos_avg = np.zeros(shape=(size, size)).tolist()
            neg_avg = np.zeros(shape=(size, size)).tolist()
            current_label = label
        
        imgA = get_image(f"/home/clenneabig/Desktop/AIML591/FirstYear/Full Dataset/{label}/{img}")

        transform = T.Compose([T.Resize(size=(384, 384))]) 

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(transform(imgA)), 
                                            batch_predict_gen(label), # classification function
                                            top_labels=5, 
                                            #labels=(label_to_idx[label],),
                                            hide_color=0,
                                            batch_size=128, 
                                            num_samples=1000)
        
        #print(explanation.local_exp)

        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        with open(f"./explanations2.0/top_label/{label}.txt", 'a') as fp:
            fp.write(f"{img[:-4]}: {idx_to_label[ind]}\n")
            fp.close()

        #print(explanation.top_labels)

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])

        for i in range(size):
            for j in range(size):
                super_pixel = explanation.segments[i][j]
                weight = dict_heatmap[super_pixel]

                if weight < 0:
                    neg_avg[i][j] += weight
                    total_neg_avg[i][j] += weight
                if weight > 0:
                    pos_avg[i][j] += weight
                    total_pos_avg[i][j] += weight

                abs_avg[i][j] += np.abs(weight)
                total_abs_avg[i][j] += np.abs(weight)
                

        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        np.save(f"./explanations2.0/heatmap/{label}/{img[:-4]}_heatmap.npy", heatmap)
        

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, negative_only=False, num_features=10, hide_rest=False)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        Image.fromarray((img_boundry1 * 255).astype(np.uint8)).save(f"./explanations2.0/border_pos/{label}/{img[:-4]}_borderpos.jpeg")
        #plt.imshow(img_boundry1)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=True, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        Image.fromarray((img_boundry2 * 255).astype(np.uint8)).save(f"./explanations2.0/border_neg/{label}/{img[:-4]}_borderneg.jpeg")

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, negative_only=False, num_features=10, hide_rest=False)
        img_boundry3 = mark_boundaries(temp/255.0, mask)
        Image.fromarray((img_boundry3 * 255).astype(np.uint8)).save(f"./explanations2.0/posneg/{label}/{img[:-4]}_posneg.jpeg")
        # plt.imshow(img_boundry2)

    label_avgs[label_to_idx[current_label]].append(abs_avg)
    label_avgs[label_to_idx[current_label]].append(pos_avg)
    label_avgs[label_to_idx[current_label]].append(neg_avg)

    np.save(f"./explanations2.0/heatmap/abs_avgheatmap.npy", np.asarray(total_abs_avg))
    np.save(f"./explanations2.0/heatmap/pos_avgheatmap.npy", np.asarray(total_pos_avg))
    np.save(f"./explanations2.0/heatmap/neg_avgheatmap.npy", np.asarray(total_neg_avg))

    for i in range(len(label_avgs)):
        label = idx_to_label[i]
        for j in range(len(label_avgs[i])):

            if j == 0:
                np.save(f"./explanations2.0/heatmap/{label}/abs_avgheatmap.npy", np.asarray(label_avgs[i][j]))
            elif j == 1:
                np.save(f"./explanations2.0/heatmap/{label}/pos_avgheatmap.npy", np.asarray(label_avgs[i][j]))
            else:
                np.save(f"./explanations2.0/heatmap/{label}/neg_avgheatmap.npy", np.asarray(label_avgs[i][j]))
