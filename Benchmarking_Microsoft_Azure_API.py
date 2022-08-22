import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

name_list = ["sur", "fea", "dis", "hap", "sad", "ang", "neu"]
new_name_list = ["sur", "fea", "dis", "hap", "sad", "ang", "neu", "non"]
emotions = ["surprise", "fear", "disgust", "happy", "sadness", "angry", "neutral"]
emotions_dic = {"surprise":0, "fear":1, "disgust":2, "happiness":3, "sadness":4, "anger":5, "neutral":6}
emotions_dic_non = {"surprise":0, "fear":1, "disgust":2, "happiness":3, "sadness":4, "anger":5, "neutral":6, "None":7}


operations = ["flipping", "rotation_without_crop_15", "rotation_without_crop_30", "rotation_without_crop_45",
              "rotation_without_crop_-15", "rotation_without_crop_-30", "rotation_without_crop_-45",
              "shifting_left_quarter", "shifting_left_half", "shifting_right_quarter", "shifting_right_half",
              "shifting_up_quarter", "shifting_up_half", "shifting_down_quarter", "shifting_down_half",
              "illumination_L1", "illumination_L2", "illumination_L3", "illumination_L4",
              "brightness_25", "brightness_50", "brightness_75",
              "brightness_-25", "brightness_-50", "brightness_-75",
              "gaussian_blur_L1", "gaussian_blur_L2", "gaussian_blur_L3",
              "gaussian_noise_10", "gaussian_noise_40", "gaussian_noise_70",
              "color_jitter_L1", "color_jitter_L2", "color_jitter_L3",
              "cycleGan_style_cezanne", "cycleGan_style_monet", "cycleGan_style_vangogh"]

flipping_path = "flipping"
rotation_without_crop_15_path = "rotation_without_crop/15"
rotation_without_crop_30_path = "rotation_without_crop/30"
rotation_without_crop_45_path = "rotation_without_crop/45"
rotation_without_crop_m15_path = "rotation_without_crop/-15"
rotation_without_crop_m30_path = "rotation_without_crop/-30"
rotation_without_crop_m45_path = "rotation_without_crop/-45"
shifting_left_quarter_path = "shifting/left_quarter"
shifting_left_half_path = "shifting/left_half"
shifting_right_quarter_path = "shifting/right_quarter"
shifting_right_half_path = "shifting/right_half"
shifting_up_quarter_path = "shifting/up_quarter"
shifting_up_half_path = "shifting/up_half"
shifting_down_quarter_path = "shifting/down_quarter"
shifting_down_half_path = "shifting/down_half"
illumination_L1_path = "illumination/L_1"
illumination_L2_path = "illumination/L_2"
illumination_L3_path = "illumination/L_3"
illumination_L4_path = "illumination/L_4"
brightness_25_path = "brightness/25"
brightness_50_path = "brightness/50"
brightness_75_path = "brightness/75"
brightness_m25_path = "brightness/-25"
brightness_m50_path = "brightness/-50"
brightness_m75_path = "brightness/-75"
gaussian_blur_L1_path = "gaussian_blur/L_1"
gaussian_blur_L2_path = "gaussian_blur/L_2"
gaussian_blur_L3_path = "gaussian_blur/L_3"
gaussian_noise_10_path = "gaussian_noise/10"
gaussian_noise_40_path = "gaussian_noise/40"
gaussian_noise_70_path = "gaussian_noise/70"
color_jitter_L1_path = "color_jitter/L_1"
color_jitter_L2_path = "color_jitter/L_2"
color_jitter_L3_path = "color_jitter/L_3"
cycleGan_style_cezanne_path = "cycleGan/style_cezanne"
cycleGan_style_monet_path = "cycleGan/style_monet"
cycleGan_style_vangogh_path = "cycleGan/style_vangogh"

operations_path = [flipping_path, rotation_without_crop_15_path, rotation_without_crop_30_path,
                   rotation_without_crop_45_path, rotation_without_crop_m15_path, rotation_without_crop_m30_path,
                   rotation_without_crop_m45_path, shifting_left_quarter_path, shifting_left_half_path,
                   shifting_right_quarter_path, shifting_right_half_path, shifting_up_quarter_path,
                   shifting_up_half_path, shifting_down_quarter_path, shifting_down_half_path,
                   illumination_L1_path, illumination_L2_path, illumination_L3_path, illumination_L4_path,
                   brightness_25_path, brightness_50_path, brightness_75_path,
                   brightness_m25_path, brightness_m50_path, brightness_m75_path,
                   gaussian_blur_L1_path, gaussian_blur_L2_path, gaussian_blur_L3_path,
                   gaussian_noise_10_path, gaussian_noise_40_path, gaussian_noise_70_path,
                   color_jitter_L1_path, color_jitter_L2_path, color_jitter_L3_path,
                   cycleGan_style_cezanne_path, cycleGan_style_monet_path, cycleGan_style_vangogh_path]


for index in range(19, 31):
    operation = operations[index]
    ope_path = operations_path[index]
    print(operation)

    # data input
    index = 0
    test_data = []
    test_labels = []

    for emotion in emotions:
        images_data = glob.glob(
            "/Users/kangning/phd/Projects/Comparisons_API/test_data_for_APIs_augmentation/%s/%s/*" % (emotion, ope_path))
        for image_path in images_data:
            test_data.append(image_path)
            test_labels.append(emotions.index(emotion))

    print(np.array(test_data).shape)
    print(np.array(test_labels).shape)

    # use API and get label_analysis
    import requests
    import json
    import time

    subscription_key = ''  # use your key here
    assert subscription_key

    face_api_url = 'https://australiaeast.api.cognitive.microsoft.com/face/v1.0/detect'

    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
        'recognitionModel': 'recognition_02',
    }

    start_time = time.time()
    label_analysis = []
    label_test = []

    for image_path in test_data:
        image = open(image_path, "rb").read()
        response = requests.post(face_api_url, params=params,
                                 headers=headers, data=image)

        analysis = response.json()
        #     print(analysis)
        label_analysis.append(analysis)
        index += 1
        print("index :", index)

    elapse_time = time.time() - start_time
    print(label_analysis)
    print(elapse_time)

    # get label_test
    label_test = []

    def label_detect(image_emotions):
        label_value = 0
        label = None
        for emo in image_emotions:
            if emo != "contempt" and image_emotions[emo] > label_value:
                label_value = image_emotions[emo]
                label = emo

        return label


    for analysis in label_analysis:
        if analysis != []:
            image_emotions = analysis[0]["faceAttributes"]["emotion"]
            #         print(image_emotions)
            label_test.append(label_detect(image_emotions))
        else:
            print("CANNOT DETECT FACE")
            label_test.append("None")

    print(label_test)

    # get accuracy
    cor = 0
    err = 0
    non = 0

    non_list = []
    err_list = []

    for i in range(np.array(label_test).shape[0]):
        if label_test[i] == "None":
            non += 1
            right_label = [key for key, value in emotions_dic.items() if value == test_labels[i]][0]
            non_pair = right_label + "__None"
            non_list.append(non_pair)
        else:
            if (emotions_dic[label_test[i]]) == test_labels[i]:
                cor += 1
            else:
                err += 1
                right_label = [key for key, value in emotions_dic.items() if value == test_labels[i]][0]
                wrong_label = label_test[i]
                err_pair = right_label + "__" + wrong_label
                err_list.append(err_pair)

    print(err)
    print(cor)
    print(non)
    print(err + cor + non)

    detect_rate = (100 * (cor + err)) / (cor + err + non)
    print("detect rate: ", detect_rate)
    if cor + err != 0:
        r_accracy = (100 * cor) / (cor + err)
        print("relative accuracy: ", r_accracy)
    o_accracy = (100 * cor) / (cor + err + non)
    print("overall accuracy: ", o_accracy)


    # save data
    with open("/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_accuracy_results/" + str(
                operation) + ".txt", "w") as f:
        f.write("elapse_time: " + str(elapse_time) + "\n")
        f.write("err: " + str(err) + "\n")
        f.write("cor: " + str(cor) + "\n")
        f.write("non: " + str(non) + "\n")
        f.write("all: " + str(err + cor + non) + "\n")
        f.write("detect rate: " + str(detect_rate) + "\n")
        f.write("overall accuracy: " + str(o_accracy) + "\n")

    file1 = open(
        "/Users/kangning/phd/Projects/Comparisons_API/API_results/Experiment_3_" + ope_path + "/Microsoft_Azure_API_label_analysis.txt",
        "w")
    for item in label_analysis:
        file1.write(str(item))
        file1.write("\n")
    file1.close()

    file2 = open(
        "/Users/kangning/phd/Projects/Comparisons_API/API_results/Experiment_3_" + ope_path + "/Microsoft_Azure_API_test_data.txt",
        "w")
    for index in range(np.array(test_data).shape[0]):
        file2.write(str(test_data[index]) + " " + str(test_labels[index]))
        file2.write("\n")
    file2.close()

    file3 = open(
        "/Users/kangning/phd/Projects/Comparisons_API/API_results/Experiment_3_" + ope_path + "/Microsoft_Azure_API_label_test.txt",
        "w")
    for item in label_test:
        file3.write(item)
        file3.write("\n")
    file3.close()

    file4 = open(
        "/Users/kangning/phd/Projects/Comparisons_API/API_results/Experiment_3_" + ope_path + "/Microsoft_Azure_API_err_list.txt", "w")
    for err_pair in err_list:
        file4.write(err_pair)
        file4.write("\n")
    file4.close()

    if non != 0:
        file5 = open(
            "/Users/kangning/phd/Projects/Comparisons_API/API_results/Experiment_3_" + ope_path + "/Microsoft_Azure_API_non_list.txt",
            "w")
        for non_pair in non_list:
            file5.write(non_pair)
            file5.write("\n")
        file5.close()

    # analysis err_list/non_list number
    sur_err_num = 0
    fea_err_num = 0
    dis_err_num = 0
    hap_err_num = 0
    sad_err_num = 0
    ang_err_num = 0
    neu_err_num = 0

    for err_pair in err_list:
        if err_pair.split("__")[0] == "surprise":
            sur_err_num += 1
        elif err_pair.split("__")[0] == "fear":
            fea_err_num += 1
        elif err_pair.split("__")[0] == "disgust":
            dis_err_num += 1
        elif err_pair.split("__")[0] == "happiness":
            hap_err_num += 1
        elif err_pair.split("__")[0] == "sadness":
            sad_err_num += 1
        elif err_pair.split("__")[0] == "anger":
            ang_err_num += 1
        elif err_pair.split("__")[0] == "neutral":
            neu_err_num += 1

    print(sur_err_num)
    print(fea_err_num)
    print(dis_err_num)
    print(hap_err_num)
    print(sad_err_num)
    print(ang_err_num)
    print(neu_err_num)

    print(sur_err_num + fea_err_num + dis_err_num + hap_err_num +
          sad_err_num + ang_err_num + neu_err_num)

    sur_non_num = 0
    fea_non_num = 0
    dis_non_num = 0
    hap_non_num = 0
    sad_non_num = 0
    ang_non_num = 0
    neu_non_num = 0

    for non_pair in non_list:
        if non_pair.split("__")[0] == "surprise":
            sur_non_num += 1
        elif non_pair.split("__")[0] == "fear":
            fea_non_num += 1
        elif non_pair.split("__")[0] == "disgust":
            dis_non_num += 1
        elif non_pair.split("__")[0] == "happiness":
            hap_non_num += 1
        elif non_pair.split("__")[0] == "sadness":
            sad_non_num += 1
        elif non_pair.split("__")[0] == "anger":
            ang_non_num += 1
        elif non_pair.split("__")[0] == "neutral":
            neu_non_num += 1

    print(sur_non_num)
    print(fea_non_num)
    print(dis_non_num)
    print(hap_non_num)
    print(sad_non_num)
    print(ang_non_num)
    print(neu_non_num)

    print(sur_non_num + fea_non_num + dis_non_num +
          hap_non_num + sad_non_num + ang_non_num + neu_non_num)

    sur_num = 0
    fea_num = 0
    dis_num = 0
    hap_num = 0
    sad_num = 0
    ang_num = 0
    neu_num = 0

    for item in test_labels:
        if item == 0:
            sur_num += 1
        elif item == 1:
            fea_num += 1
        elif item == 2:
            dis_num += 1
        elif item == 3:
            hap_num += 1
        elif item == 4:
            sad_num += 1
        elif item == 5:
            ang_num += 1
        elif item == 6:
            neu_num += 1

    print(sur_num)
    print(fea_num)
    print(dis_num)
    print(hap_num)
    print(sad_num)
    print(ang_num)
    print(neu_num)

    print(sur_num + fea_num + dis_num + hap_num +
          sad_num + ang_num + neu_num)

    with open("/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_accuracy_results/" + str(
                operation) + "_analysis.txt", "w") as f1:

        f1.write(str(sur_err_num) + "\n")
        f1.write(str(fea_err_num) + "\n")
        f1.write(str(dis_err_num) + "\n")
        f1.write(str(hap_err_num) + "\n")
        f1.write(str(sad_err_num) + "\n")
        f1.write(str(ang_err_num) + "\n")
        f1.write(str(neu_err_num) + "\n")
        f1.write(
            str(sur_err_num + fea_err_num + dis_err_num + hap_err_num + sad_err_num + ang_err_num + neu_err_num) + "\n")
        f1.write("\n")

        if non != 0:
            f1.write(str(sur_non_num) + "\n")
            f1.write(str(fea_non_num) + "\n")
            f1.write(str(dis_non_num) + "\n")
            f1.write(str(hap_non_num) + "\n")
            f1.write(str(sad_non_num) + "\n")
            f1.write(str(ang_non_num) + "\n")
            f1.write(str(neu_non_num) + "\n")
            f1.write(
                str(sur_non_num + fea_non_num + dis_non_num + hap_non_num + sad_non_num + ang_non_num + neu_non_num) + "\n")
            f1.write("\n")

        f1.write(str(sur_num) + "\n")
        f1.write(str(fea_num) + "\n")
        f1.write(str(dis_num) + "\n")
        f1.write(str(hap_num) + "\n")
        f1.write(str(sad_num) + "\n")
        f1.write(str(ang_num) + "\n")
        f1.write(str(neu_num) + "\n")
        f1.write(
            str(sur_num + fea_num + dis_num + hap_num + sad_num + ang_num + neu_num) + "\n")
        f1.write("\n")

    # draw pictures
    if non != 0:
        # barplot
        emotion_num_list = [sur_num, fea_num, dis_num, hap_num,
                            sad_num, ang_num, neu_num]

        emotion_not_non_num_list = [sur_num - sur_non_num, fea_num - fea_non_num, dis_num - dis_non_num,
                                    hap_num - hap_non_num, sad_num - sad_non_num, ang_num - ang_non_num,
                                    neu_num - neu_non_num]

        emotion_err_num_list = [sur_err_num, fea_err_num, dis_err_num, hap_err_num,
                                sad_err_num, ang_err_num, neu_err_num]

        emotion_cor_num_list = [sur_num - sur_non_num - sur_err_num, fea_num - fea_non_num - fea_err_num,
                                dis_num - dis_non_num - dis_err_num, hap_num - hap_non_num - hap_err_num,
                                sad_num - sad_non_num - sad_err_num, ang_num - ang_non_num - ang_err_num,
                                neu_num - neu_non_num - neu_err_num]

        sns.set_context({"figure.figsize": (5, 4)})
        sns.barplot(x=name_list, y=emotion_num_list, color="yellow")
        sns.barplot(x=name_list, y=emotion_not_non_num_list, color="red")
        bottom_plot = sns.barplot(x=name_list, y=emotion_cor_num_list, color="blue")
        bottom_plot.set_ylim(0, 160)

        topbar = plt.Rectangle((0, 0), 1, 1, fc="yellow", edgecolor='none')
        middlebar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
        bottombar = plt.Rectangle((0, 0), 1, 1, fc="blue", edgecolor='none')
        l = plt.legend([topbar, middlebar, bottombar], ['None', 'Error', 'Correct'], loc='upper right')
        l.draw_frame(True)

        plt.savefig(
            "/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_results_pictures/" + operation + "_barplot.png")
        plt.close()

        # heatmap
        heat_list = [[0] * 8 for i in range(7)]

        for err_pair in err_list:
            row = int(emotions_dic_non[err_pair.split("__")[0]])  # correct
            col = int(emotions_dic_non[err_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for non_pair in non_list:
            row = int(emotions_dic_non[non_pair.split("__")[0]])  # correct
            col = int(emotions_dic_non[non_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for cor_index in range(7):
            heat_list[cor_index][cor_index] = emotion_cor_num_list[cor_index]

        # print(heat_list)
        with open("/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_accuracy_results/" + str(
                operation) + "_heat_list.txt", "w") as f2:
            for i in range(7):
                f2.write(str(heat_list[i]) + "\n")

        heat_list = np.array(heat_list).astype("float32")
        for i in range(7):
            heat_list[i] /= emotion_num_list[i]

        print(heat_list)

        sns.set()
        ax = sns.heatmap(heat_list, cmap="Blues", xticklabels=new_name_list, yticklabels=name_list, square=True)

        plt.savefig(
            "/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_results_pictures/" + operation + "_heatmap.png")
        plt.close()

    else:
        # barplot
        emotion_num_list = [sur_num, fea_num, dis_num, hap_num,
                            sad_num, ang_num, neu_num]

        emotion_cor_num_list = [sur_num - sur_err_num, fea_num - fea_err_num, dis_num - dis_err_num,
                                hap_num - hap_err_num, sad_num - sad_err_num,
                                ang_num - ang_err_num, neu_num - neu_err_num]

        emotion_err_num_list = [sur_err_num, fea_err_num, dis_err_num, hap_err_num,
                                sad_err_num, ang_err_num, neu_err_num]

        sns.set_context({"figure.figsize": (5, 4)})
        sns.barplot(x=name_list, y=emotion_num_list, color="red")
        bottom_plot = sns.barplot(x=name_list, y=emotion_cor_num_list, color="blue")
        bottom_plot.set_ylim(0, 160)

        topbar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
        bottombar = plt.Rectangle((0, 0), 1, 1, fc="blue", edgecolor='none')
        l = plt.legend([topbar, bottombar], ['Error', 'Correct'], loc='upper right')
        l.draw_frame(True)

        plt.savefig(
            "/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_results_pictures/" + operation + "_barplot.png")
        plt.close()

        # heatmap
        heat_list = [[0] * 7 for i in range(7)]
        for err_pair in err_list:
            row = int(emotions_dic[err_pair.split("__")[0]])  # correct
            col = int(emotions_dic[err_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for cor_index in range(7):
            heat_list[cor_index][cor_index] = emotion_cor_num_list[cor_index]

        # print(heat_list)
        with open("/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_accuracy_results/" + str(
                operation) + "_heat_list.txt", "w") as f3:
            for i in range(7):
                f3.write(str(heat_list[i]) + "\n")

        heat_list = np.array(heat_list).astype("float32")
        for i in range(7):
            heat_list[i] /= emotion_num_list[i]

        print(heat_list)

        sns.set()
        ax = sns.heatmap(heat_list, cmap="Blues", xticklabels=name_list, yticklabels=name_list, square=True)

        plt.savefig(
            "/Users/kangning/phd/Projects/Comparisons_API/report/Experiment_3/Microsoft_Azure_API_results_pictures/" + operation + "_heatmap.png")
        plt.close()



