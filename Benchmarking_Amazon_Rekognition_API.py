import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

name_list = ["sur", "dis", "hap", "sad", "ang", "neu"]
new_name_list = ["sur", "dis", "hap", "sad", "ang", "neu", "non"]
emotions = ["surprise", "disgust", "happy", "sadness", "angry", "neutral"]
emotions_dic = {"SURPRISED":0, "DISGUSTED":1, "HAPPY":2, "SAD":3, "ANGRY":4, "CALM":5}
emotions_dic_non = {"SURPRISED":0, "DISGUSTED":1, "HAPPY":2, "SAD":3, "ANGRY":4, "CALM":5, "None":6}


operations = ["flipping", "rotation_without_crop_15", "rotation_without_crop_30", "rotation_without_crop_45",
              "rotation_without_crop_-15", "rotation_without_crop_-30", "rotation_without_crop_-45",
              "shifting_left_quarter", "shifting_left_half", "shifting_right_quarter", "shifting_right_half",
              "shifting_up_quarter", "shifting_up_half", "shifting_down_quarter", "shifting_down_half",
              "illumination_L1", "illumination_L2", "illumination_L3", "gaussian_blur_L1", "gaussian_blur_L2",
              "gaussian_blur_L3", "gaussian_noise_L1", "gaussian_noise_L2", "gaussian_noise_L3",
              "color_jitter_L1", "color_jitter_L2", "color_jitter_L3", "cycleGan_style_cezanne",
              "cycleGan_style_monet", "cycleGan_style_vangogh"]

flipping_path = "flipping"
rotation_without_crop_15_path = "rotation_without_crop\\15"
rotation_without_crop_30_path = "rotation_without_crop\\30"
rotation_without_crop_45_path = "rotation_without_crop\\45"
rotation_without_crop_m15_path = "rotation_without_crop\\-15"
rotation_without_crop_m30_path = "rotation_without_crop\\-30"
rotation_without_crop_m45_path = "rotation_without_crop\\-45"
shifting_left_quarter_path = "shifting\\left_quarter"
shifting_left_half_path = "shifting\\left_half"
shifting_right_quarter_path = "shifting\\right_quarter"
shifting_right_half_path = "shifting\\right_half"
shifting_up_quarter_path = "shifting\\up_quarter"
shifting_up_half_path = "shifting\\up_half"
shifting_down_quarter_path = "shifting\\down_quarter"
shifting_down_half_path = "shifting\\down_half"
illumination_L1_path = "illumination\\L_1"
illumination_L2_path = "illumination\\L_2"
illumination_L3_path = "illumination\\L_3"
gaussian_blur_L1_path = "gaussian_blur\\L_1"
gaussian_blur_L2_path = "gaussian_blur\\L_2"
gaussian_blur_L3_path = "gaussian_blur\\L_3"
gaussian_noise_L1_path = "gaussian_noise\\L_1"
gaussian_noise_L2_path = "gaussian_noise\\L_2"
gaussian_noise_L3_path = "gaussian_noise\\L_3"
color_jitter_L1_path = "color_jitter\\L_1"
color_jitter_L2_path = "color_jitter\\L_2"
color_jitter_L3_path = "color_jitter\\L_3"
cycleGan_style_cezanne_path = "cycleGan\\style_cezanne"
cycleGan_style_monet_path = "cycleGan\\style_monet"
cycleGan_style_vangogh_path = "cycleGan\\style_vangogh"

operations_path = [flipping_path, rotation_without_crop_15_path, rotation_without_crop_30_path,
                   rotation_without_crop_45_path, rotation_without_crop_m15_path, rotation_without_crop_m30_path,
                   rotation_without_crop_m45_path, shifting_left_quarter_path, shifting_left_half_path,
                   shifting_right_quarter_path, shifting_right_half_path, shifting_up_quarter_path,
                   shifting_up_half_path, shifting_down_quarter_path, shifting_down_half_path,
                   illumination_L1_path, illumination_L2_path, illumination_L3_path,
                   gaussian_blur_L1_path, gaussian_blur_L2_path, gaussian_blur_L3_path,
                   gaussian_noise_L1_path, gaussian_noise_L2_path, gaussian_noise_L3_path,
                   color_jitter_L1_path, color_jitter_L2_path, color_jitter_L3_path,
                   cycleGan_style_cezanne_path, cycleGan_style_monet_path, cycleGan_style_vangogh_path]


for index in range(30):
    operation = operations[index]
    ope_path = operations_path[index]
    print(operation)

    # data input
    test_data = []
    test_labels = []

    for emotion in emotions:
        images_data = glob.glob(
            "D:\\phd\\code\\Project1\\Study1_database\\test_data_for_APIs_augmentation\\%s\\%s\\*" % (emotion, ope_path))
        for image_path in images_data:
            test_data.append(image_path)
            test_labels.append(emotions.index(emotion))

    print(np.array(test_data).shape)
    print(np.array(test_labels).shape)

    # use API and get label_analysis
    import time
    import boto3

    client = boto3.client('rekognition')

    label_analysis = []

    start_time = time.time()
    for image_path in test_data:
        with open(image_path, 'rb') as image:
            response = client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
            label_analysis.append(response["FaceDetails"][0]["Emotions"])
    # print(response["FaceDetails"][0]["Emotions"])

    elapse_time = time.time() - start_time

    print(elapse_time)
    print(label_analysis)

    # get label_test
    label_test = []

    def label_detect(analysis):
        label_value = 0
        label = None
        for emotions in analysis:
            if emotions["Type"] != "CONFUSED" and emotions["Confidence"] > label_value:
                label_value = emotions["Confidence"]
                label = emotions["Type"]
        return label


    for analysis in label_analysis:
        if analysis != []:
            label_test.append(label_detect(analysis))
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
    r_accracy = (100 * cor) / (cor + err)
    print("relative accuracy: ", r_accracy)
    o_accracy = (100 * cor) / (cor + err + non)
    print("overall accuracy: ", o_accracy)


    # save data
    with open("D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_accuracy_results\\" + str(
                operation) + ".txt", "w") as f:
        f.write("elapse_time: " + str(elapse_time) + "\n")
        f.write("err: " + str(err) + "\n")
        f.write("cor: " + str(cor) + "\n")
        f.write("non: " + str(non) + "\n")
        f.write("all: " + str(err + cor + non) + "\n")
        f.write("detect rate: " + str(detect_rate) + "\n")
        f.write("overall accuracy: " + str(o_accracy) + "\n")

    file1 = open(
        "D:\\phd\\code\\Project1\\API_results\\Experiment_3_" + ope_path + "\\Amazon_Rekognition_API_label_analysis.txt",
        "w")
    for item in label_analysis:
        file1.write(str(item))
        file1.write("\n")
    file1.close()

    file2 = open(
        "D:\\phd\\code\\Project1\\API_results\\Experiment_3_" + ope_path + "\\Amazon_Rekognition_API_test_data.txt",
        "w")
    for index in range(np.array(test_data).shape[0]):
        file2.write(str(test_data[index]) + " " + str(test_labels[index]))
        file2.write("\n")
    file2.close()

    file3 = open(
        "D:\\phd\\code\\Project1\\API_results\\Experiment_3_" + ope_path + "\\Amazon_Rekognition_API_label_test.txt",
        "w")
    for item in label_test:
        file3.write(item)
        file3.write("\n")
    file3.close()

    file4 = open(
        "D:\\phd\\code\\Project1\\API_results\\Experiment_3_" + ope_path + "\\Amazon_Rekognition_API_err_list.txt", "w")
    for err_pair in err_list:
        file4.write(err_pair)
        file4.write("\n")
    file4.close()

    if non != 0:
        file5 = open(
            "D:\\phd\\code\\Project1\\API_results\\Experiment_3_" + ope_path + "\\Amazon_Rekognition_API_non_list.txt",
            "w")
        for non_pair in non_list:
            file5.write(non_pair)
            file5.write("\n")
        file5.close()

    # analysis err_list/non_list number
    sur_err_num = 0
    dis_err_num = 0
    hap_err_num = 0
    sad_err_num = 0
    ang_err_num = 0
    neu_err_num = 0

    for err_pair in err_list:
        if err_pair.split("__")[0] == "SURPRISED":
            sur_err_num += 1
        elif err_pair.split("__")[0] == "DISGUSTED":
            dis_err_num += 1
        elif err_pair.split("__")[0] == "HAPPY":
            hap_err_num += 1
        elif err_pair.split("__")[0] == "SAD":
            sad_err_num += 1
        elif err_pair.split("__")[0] == "ANGRY":
            ang_err_num += 1
        elif err_pair.split("__")[0] == "CALM":
            neu_err_num += 1

    print(sur_err_num)
    print(dis_err_num)
    print(hap_err_num)
    print(sad_err_num)
    print(ang_err_num)
    print(neu_err_num)

    print(sur_err_num + dis_err_num + hap_err_num +
          sad_err_num + ang_err_num + neu_err_num)

    sur_non_num = 0
    dis_non_num = 0
    hap_non_num = 0
    sad_non_num = 0
    ang_non_num = 0
    neu_non_num = 0

    for non_pair in non_list:
        if non_pair.split("__")[0] == "SURPRISED":
            sur_non_num += 1
        elif non_pair.split("__")[0] == "DISGUSTED":
            dis_non_num += 1
        elif non_pair.split("__")[0] == "HAPPY":
            hap_non_num += 1
        elif non_pair.split("__")[0] == "SAD":
            sad_non_num += 1
        elif non_pair.split("__")[0] == "ANGRY":
            ang_non_num += 1
        elif non_pair.split("__")[0] == "CALM":
            neu_non_num += 1

    print(sur_non_num)
    print(dis_non_num)
    print(hap_non_num)
    print(sad_non_num)
    print(ang_non_num)
    print(neu_non_num)

    print(sur_non_num + dis_non_num + hap_non_num +
          sad_non_num + ang_non_num + neu_non_num)

    sur_num = 0
    dis_num = 0
    hap_num = 0
    sad_num = 0
    ang_num = 0
    neu_num = 0

    for item in test_labels:
        if item == 0:
            sur_num += 1
        elif item == 1:
            dis_num += 1
        elif item == 2:
            hap_num += 1
        elif item == 3:
            sad_num += 1
        elif item == 4:
            ang_num += 1
        elif item == 5:
            neu_num += 1

    print(sur_num)
    print(dis_num)
    print(hap_num)
    print(sad_num)
    print(ang_num)
    print(neu_num)

    print(sur_num + dis_num + hap_num +
          sad_num + ang_num + neu_num)

    with open("D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_accuracy_results\\" + str(
                operation) + "_analysis.txt", "w") as f1:

        f1.write(str(sur_err_num) + "\n")
        f1.write(str(dis_err_num) + "\n")
        f1.write(str(hap_err_num) + "\n")
        f1.write(str(sad_err_num) + "\n")
        f1.write(str(ang_err_num) + "\n")
        f1.write(str(neu_err_num) + "\n")
        f1.write(
            str(sur_err_num + dis_err_num + hap_err_num + sad_err_num + ang_err_num + neu_err_num) + "\n")
        f1.write("\n")

        if non != 0:
            f1.write(str(sur_non_num) + "\n")
            f1.write(str(dis_non_num) + "\n")
            f1.write(str(hap_non_num) + "\n")
            f1.write(str(sad_non_num) + "\n")
            f1.write(str(ang_non_num) + "\n")
            f1.write(str(neu_non_num) + "\n")
            f1.write(
                str(sur_non_num + dis_non_num + hap_non_num + sad_non_num + ang_non_num + neu_non_num) + "\n")
            f1.write("\n")

        f1.write(str(sur_num) + "\n")
        f1.write(str(dis_num) + "\n")
        f1.write(str(hap_num) + "\n")
        f1.write(str(sad_num) + "\n")
        f1.write(str(ang_num) + "\n")
        f1.write(str(neu_num) + "\n")
        f1.write(
            str(sur_num + dis_num + hap_num + sad_num + ang_num + neu_num) + "\n")
        f1.write("\n")

    # draw pictures
    if non != 0:
        # barplot
        emotion_num_list = [sur_num, dis_num, hap_num,
                            sad_num, ang_num, neu_num]

        emotion_not_non_num_list = [sur_num - sur_non_num, dis_num - dis_non_num,
                                    hap_num - hap_non_num, sad_num - sad_non_num,
                                    ang_num - ang_non_num, neu_num - neu_non_num]

        emotion_err_num_list = [sur_err_num, dis_err_num, hap_err_num,
                                sad_err_num, ang_err_num, neu_err_num]

        emotion_cor_num_list = [sur_num - sur_non_num - sur_err_num, dis_num - dis_non_num - dis_err_num,
                                hap_num - hap_non_num - hap_err_num, sad_num - sad_non_num - sad_err_num,
                                ang_num - ang_non_num - ang_err_num, neu_num - neu_non_num - neu_err_num]

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
            "D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_results_pictures\\" + operation + "_barplot.png")
        plt.close()

        # heatmap
        heat_list = [[0] * 7 for i in range(6)]

        for err_pair in err_list:
            row = int(emotions_dic_non[err_pair.split("__")[0]])  # correct
            col = int(emotions_dic_non[err_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for non_pair in non_list:
            row = int(emotions_dic_non[non_pair.split("__")[0]])  # correct
            col = int(emotions_dic_non[non_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for cor_index in range(6):
            heat_list[cor_index][cor_index] = emotion_cor_num_list[cor_index]

        # print(heat_list)
        with open("D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_accuracy_results\\" + str(
                operation) + "_heat_list.txt", "w") as f2:
            for i in range(6):
                f2.write(str(heat_list[i]) + "\n")

        heat_list = np.array(heat_list).astype("float32")
        for i in range(6):
            heat_list[i] /= emotion_num_list[i]

        print(heat_list)

        sns.set()
        ax = sns.heatmap(heat_list, cmap="Blues", xticklabels=new_name_list, yticklabels=name_list, square=True)

        plt.savefig(
            "D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_results_pictures\\" + operation + "_heatmap.png")
        plt.close()

    else:
        # barplot
        emotion_num_list = [sur_num, dis_num, hap_num,
                            sad_num, ang_num, neu_num]

        emotion_cor_num_list = [sur_num - sur_err_num, dis_num - dis_err_num,
                                hap_num - hap_err_num, sad_num - sad_err_num,
                                ang_num - ang_err_num, neu_num - neu_err_num]

        emotion_err_num_list = [sur_err_num, dis_err_num, hap_err_num,
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
            "D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_results_pictures\\" + operation + "_barplot.png")
        plt.close()

        # heatmap
        heat_list = [[0] * 6 for i in range(6)]
        for err_pair in err_list:
            row = int(emotions_dic[err_pair.split("__")[0]])  # correct
            col = int(emotions_dic[err_pair.split("__")[-1]])  # error
            heat_list[row][col] += 1

        for cor_index in range(6):
            heat_list[cor_index][cor_index] = emotion_cor_num_list[cor_index]

        # print(heat_list)
        with open("D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_accuracy_results\\" + str(
                operation) + "_heat_list.txt", "w") as f3:
            for i in range(6):
                f3.write(str(heat_list[i]) + "\n")

        heat_list = np.array(heat_list).astype("float32")
        for i in range(6):
            heat_list[i] /= emotion_num_list[i]

        print(heat_list)

        sns.set()
        ax = sns.heatmap(heat_list, cmap="Blues", xticklabels=name_list, yticklabels=name_list, square=True)

        plt.savefig(
            "D:\\phd\\code\\Project1\\report\\Experiment_3\\Amazon_Rekognition_API_results_pictures\\" + operation + "_heatmap.png")
        plt.close()



