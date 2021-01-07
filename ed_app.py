from __future__ import annotations
import json
import os
from pickle import LIST
import urllib

import cv2
import numpy as np
# import altair as alt
import pandas as pd
from pandas.core import frame
import streamlit as st
from jaitool.draw import draw_bbox
from jaitool.inference import D2Inferer
from jaitool.structures.bbox import BBox
from pyjeasy.image_utils import show_image
from seaborn import color_palette
from typing import List
import printj
# import streamlit_theme as stt

# Streamlit encourages well-structured code, like starting execution in a main() function.
# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

# External files to download.
EXTERNAL_DEPENDENCIES = {
    # "yolov3.weights": {
    #     "url": "https://pjreddie.com/media/files/yolov3.weights",
    #     "size": 248007048
    # },
    # "yolov3.cfg": {
    #     "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    #     "size": 8342
    # }
}

IMG_DATA_PATH = "data/annotated_data"
IMG_EXTENTIONS = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
IMG_NAMES = sorted([fn for fn in os.listdir(IMG_DATA_PATH)
                    if any(fn.endswith(ext) for ext in IMG_EXTENTIONS)])
CLASS_NAMES = ['bracket', 'distribution_board', 'distributor', 'downlight',
               'interphone', 'light', 'outlet', 'plumbing', 'spotlight', 'wiring']
CLASS_NAMES_OCR = ['four', 'lan', 'one_ground',
                   'six', 'tv', 'two_ground', 'two', 'wp']
RENAME2 = {
    'four':'4', 
    'lan':'LAN', 
    'one_ground':'1E',
    'six':'6', 
    'tv':'TV', 
    'two_ground':'2E', 
    'two':'2', 
    'wp':'WP'
    }
# PALETTES = np.array(color_palette(
#     # palette='Set2',
#     palette='hls',
#     n_colors=len(CLASS_NAMES)+1))*255
# RENAME = {
#     '' : 'bracket',
#     '' : 'distribution_board',
#     '' : 'distributor',
#     '' : 'downlight',
#     '' : 'interphone',
#     '' : 'light',
#     't2' : 'outlet',
#     '' : 'plumbing',
#     '' : 'spotlight',
#     '' : 'wiring'
# }


class ClassInfo:
    def __init__(self, name: str, pre_count: int = 0, gt_count: int = 0, bbox=None):
        self.name = name
        self.pre_count = pre_count
        self.gt_count = gt_count
        self.bbox = bbox


class Frame:
    def __init__(self, frame_name, class_list: LIST[str]):
        self.frame_name = frame_name
        self.classes = dict()
        for _class in class_list:
            self.classes[_class] = ClassInfo(name=_class)
        self.predict_dict = dict()


class Summary:
    def __init__(self, frame_list, class_list):
        self.frame = dict()
        for frame in frame_list:
            frame_data = Frame(frame, class_list)
            self.frame[frame] = frame_data


class STED:

    def __init__(self, data_dir: str, image_names: List[str], class_names: List[str], class_names2: List[str]):
        self.data_dir = data_dir
        self.image_names = image_names
        self.class_names = class_names
        self.class_names2 = class_names2
        self.use_ocr = False
        self.pre_count = dict()
        self.paletts = np.array(color_palette(
            # palette='Set2',
            palette='hls',
            n_colors=len(self.class_names)+1))*255
        self.summary = Summary(self.image_names, self.class_names)
        for img_name in self.image_names:
            gt_path = os.path.abspath(
                f'{self.data_dir}/{img_name.split(".")[0]}.json')
            if os.path.exists(gt_path):
                with open(gt_path) as json_file:
                    self.gt_data = json.load(json_file)
                for shape in self.gt_data["shapes"]:
                    for c in self.class_names:
                        if c == shape["label"]:
                            self.summary.frame[img_name].classes[c].gt_count += 1
                            # print(frame.classes[c].gt_count)

            # image_url = os.path.join(self.data_dir, img_name)
            # image = self.load_image(image_url)
            # self.summary.frame[img_name].predict_dict = self.rcnn(image)

        st.title('Electrical Drawing Analyser')
        # Render the readme as markdown using st.markdown.
        readme_text = st.markdown(
            self.get_file_content_as_string("instructions.md"))

        # Download external dependencies.
        for filename in EXTERNAL_DEPENDENCIES.keys():
            self.download_file(filename)

        # Once we have the dependencies, add a selector for the app mode on the sidebar.
        # st.sidebar.title("What to do")
        # app_mode = st.sidebar.selectbox("Choose the app mode",
        #     ["Show instructions", "Run the app", "Show the source code"])
        # if app_mode == "Show instructions":
        #     st.sidebar.success('To continue select "Run the app".')
        # elif app_mode == "Show the source code":
        #     readme_text.empty()
        #     st.code(get_file_content_as_string("streamlit_app.py"))
        # elif app_mode == "Run the app":
        #     readme_text.empty()
        #     run_the_app()
        readme_text.empty()
        self.run_the_app()
    # This file downloader demonstrates Streamlit animation.

    def download_file(self, file_path):
        # Don't download the file twice. (If possible, verify the download using the file length.)
        if os.path.exists(file_path):
            if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
                return
            elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
                return

        # These are handles to two visual elements to animate.
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % file_path)
            progress_bar = st.progress(0)
            with open(file_path, "wb") as output_file:
                with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                    length = int(response.info()["Content-Length"])
                    pre_counter = 0.0
                    MEGABYTES = 2.0 ** 20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        pre_counter += len(data)
                        output_file.write(data)

                        # We perform animation by overwriting the elements.
                        weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                                                (file_path, pre_counter / MEGABYTES, length / MEGABYTES))
                        progress_bar.progress(min(pre_counter / length, 1.0))

        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()

    # This is the main app app itself, which appears when the user selects "Run the app".
    def run_the_app(self, ):
        # To make Streamlit fast, st.cache allows us to reuse computation across runs.
        # In this common pattern, we download data from an endpoint only once.
        # @st.cache
        # def load_metadata(url):
        #     return pd.read_csv(url)

        # # This function uses some Pandas magic to summarize the metadata Dataframe.
        # @st.cache
        # def create_summary(metadata):
        #     one_hot_encoded = pd.get_dummies(
        #         metadata[["frame", "label"]], columns=["label"])
        #     summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
        #         "label_biker": "biker",
        #         "label_car": "car",
        #         "label_pedestrian": "pedestrian",
        #         "label_trafficLight": "traffic light",
        #         "label_truck": "truck"
        #     })
        #     return summary

        # # An amazing property of st.cached functions is that you can pipe them into
        # # one another to form a computation DAG (directed acyclic graph). Streamlit
        # # recomputes only whatever subset is required to get the right answer!
        # metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
        # summary = create_summary(metadata)

        # Uncomment these lines to peek at these DataFrames.
        # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

        # Draw the UI elements to search for objects (pedestrians, cars, etc.)
        self.selected_frame_index, self.selected_frame = self.frame_selector_ui()
        if self.selected_frame_index == None:
            st.error(
                "No frames fit the criteria. Please select different label or number.")
            return

        # Load the image from S3.
        image_url = os.path.join(self.data_dir, self.selected_frame)
        image = self.load_image(image_url)

        # Add boxes for objects on the image. These are the boxes for the ground image.
        # boxes = metadata[metadata.frame == self.selected_frame].drop(columns=[
        #                                                         "frame"])
        # draw_image_with_boxes(image, boxes, "Ground Truth",
        #     f"**Human-annotated data** (frame `{selected_frame_index}`, `{self.image_names[selected_frame_index]}`)", confidence_threshold)

        # Get the boxes for the objects detected by YOLO by running the YOLO model.
        # yolo_boxes = yolo_v3(image, confidence_threshold, overlap_threshold)
        predict_dict = self.rcnn(image)
        # predict_dict = self.summary.frame[self.selected_frame].predict_dict
        
        # Draw the UI element to select parameters for the YOLO object detector.
        self.confidence_threshold = self.object_detector_ui()
        self.selected_classes = self.class_selector_ui()
        st.sidebar.text(" \n")
        # st.sidebar.text(" \n")
        # st.sidebar.text(" \n")
        use_ocr = self.ocr_switch_ui()
        if use_ocr == "With OCR":
            self.use_ocr = True
            st.sidebar.markdown("# OCR Model")
            self.confidence_threshold2 = st.sidebar.slider(
                "Confidence threshold (OCR)", 0.0, 1.0, 0.5, 0.01)
            st.sidebar.write("Confidence threshold (OCR) is",  self.confidence_threshold2)
        else:
            self.use_ocr = False
        self.draw_image_with_boxes(image, predict_dict, f"Real-time Computer Vision",
                                   f"**Cascade-RCNN Model** (confidence `{self.confidence_threshold}`) (frame_name `{self.selected_frame}`)",
                                   self.confidence_threshold, self.selected_classes, self.use_ocr)
        self.draw_legends(self.selected_classes)
    # This sidebar UI is a little search engine to find certain object types.

    def frame_selector_ui(self):
        st.sidebar.markdown("# Frame")

        # The user can pick which type of object to search for.
        # object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)
        # The user can select a range for how many of the selected objecgt should be present.
        # min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [10, 20])
        # selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
        # len_selected_frames = len(selected_frames)
        len_selected_frames = len(self.image_names)
        if len_selected_frames < 1:
            return None, None

        # Choose a frame out of the selected frames.
        selected_frame_index = st.sidebar.slider(
            "Choose a frame (index)", 0, len_selected_frames - 1, 0)

        # Draw an altair chart in the sidebar with information on the frame.
        # objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
        # chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        #     alt.X("index:Q", scale=alt.Scale(nice=False)),
        #     alt.Y("%s:Q" % object_type))
        # selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
        # vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x = "selected_frame")
        # st.sidebar.altair_chart(alt.layer(chart, vline))

        # selected_frame = selected_frames[selected_frame_index]
        selected_frame = self.image_names[selected_frame_index]
        # ALL_LAYERS = {
        #     "t2": 0,
        #     "t10": 0,
        # }
        return selected_frame_index, selected_frame

    def class_selector_ui(self):
        st.sidebar.markdown('### Select classes')
        selected_classes = [
            class_name for class_name in self.class_names
            if st.sidebar.checkbox(class_name, True)]
        # layer for layer_name, layer in ALL_LAYERS.items()
        # if st.sidebar.checkbox(layer_name, True)]
        return selected_classes

    def ocr_switch_ui(self):
        # st.sidebar.markdown('### Select classes')
        # self.use_ocr = [
        #     class_name for class_name in ['Use OCR']
        #     if st.sidebar.checkbox(class_name, False)]
        use_ocr = st.sidebar.radio(
            "Select the Model", ["Simple", "With OCR"]
        )
        return use_ocr

    # Select frames based on the selection in the sidebar
    @staticmethod
    @st.cache(hash_funcs={np.ufunc: str})
    def get_selected_frames(summary, label, min_elts, max_elts):
        return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

    # This sidebar UI lets the user select parameters for the YOLO object detector.
    @staticmethod
    def object_detector_ui():
        st.sidebar.markdown("# Model")
        confidence_threshold = st.sidebar.slider(
            "Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        st.sidebar.write("Confidence threshold is",  confidence_threshold)
        # confidence_threshold2 = st.sidebar.slider(
        #     "Confidence threshold (OCR)", 0.0, 1.0, 0.5, 0.01)
        # overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.0, 0.01)
        return confidence_threshold  # , confidence_threshold2

    # Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
    def draw_legends(self, selected_classes):
        for selected_class in selected_classes:
            cat_id = self.class_names.index(selected_class)
            color_bbox = self.paletts[cat_id]
            rec_size = 40
            rec = np.zeros([rec_size, rec_size, 3], dtype=np.uint8)
            rec[:, :] = tuple(color_bbox)
            gt_count = self.summary.frame[self.selected_frame].classes[selected_class].gt_count
            text = f'{selected_class} (pred {self.pre_count[selected_class]}) (gt {gt_count})'
            rec = cv2.copyMakeBorder(src=rec, top=0, bottom=0, left=0, right=int(
                rec_size*len(text)*1.1), borderType=0, value=[255]*3)
            cv2.putText(img=rec, text=text, org=(int(rec_size*1.1), int(rec_size*.8)),
                        fontFace=2, fontScale=1, color=0, thickness=1, lineType=1)
            st.image(image=rec.astype(np.uint8))

    def draw_image_with_boxes(self, image, predict_dict, header, description, confidence_threshold, selected_classes, use_ocr):
        output = image.copy()
        score_list = predict_dict['score_list']
        bbox_list = predict_dict['bbox_list']
        pred_class_list = predict_dict['pred_class_list']
        pred_masks_list = predict_dict['pred_masks_list']
        pred_keypoints_list = predict_dict['pred_keypoints_list']
        vis_keypoints_list = predict_dict['vis_keypoints_list']
        kpt_confidences_list = predict_dict['kpt_confidences_list']
        for _class in self.class_names:
            self.pre_count[_class] = 0
        for score, pred_class, bbox, mask, keypoints, vis_keypoints, kpt_confidences in zip(score_list,
                                                                                            pred_class_list,
                                                                                            bbox_list,
                                                                                            pred_masks_list,
                                                                                            pred_keypoints_list,
                                                                                            vis_keypoints_list,
                                                                                            kpt_confidences_list):
            if score > confidence_threshold:
                if pred_class in selected_classes:
                    self.pre_count[pred_class] += 1
                    cat_id = self.class_names.index(pred_class)
                    color_bbox = self.paletts[cat_id]
                    output = draw_bbox(img=output, bbox=bbox, color=color_bbox,
                                       show_label=False, text=None, thickness=5)
                    # cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(
                    # xmax, ymax), color=color_bbox, thickness=2)
                    if use_ocr:
                        # if pred_class in ["outlet", "plumbing"]:
                        #     crop_box = bbox
                        #     st.image(image=output[crop_box.ymin: crop_box.ymax, crop_box.xmin: crop_box.xmax, :].astype(np.uint8),)
                        output = self.rcnn_cropped(image, pred_class, bbox, output)
                    # if pred_class in ["outlet", "plumbing"]:
                    #     crop_box = bbox.pad(pad_left = 50, pad_right = 50, pad_top = 50, pad_bottom = 50)

        # Draw the header and image.
        st.subheader(header)
        st.markdown(description)
        st.image(image=output.astype(np.uint8),
                  width=2000,
                #  use_column_width=True
                 )
        return output

    # def draw_image_with_boxes(self, image, boxes, header, description):
    #     # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    #     LABEL_COLORS = {
    #         "car": [255, 0, 0],
    #         "pedestrian": [0, 255, 0],
    #         "truck": [0, 0, 255],
    #         "trafficLight": [255, 255, 0],
    #         "biker": [255, 0, 255],
    #     }
    #     image_with_boxes = image.astype(np.float64)
    #     for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
    #         image_with_boxes[int(ymin):int(ymax), int(
    #             xmin):int(xmax), :] += LABEL_COLORS[label]
    #         image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    #     # Draw the header and image.
    #     st.subheader(header)
    #     st.markdown(description)
    #     st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

    # Download a single file and make its content available as a string.
    @staticmethod
    @st.cache(show_spinner=False)
    def get_file_content_as_string(path):
        url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
        response = urllib.request.urlopen(url)
        return response.read().decode("utf-8")

    # This function loads an image from Streamlit public repo on S3. We use st.cache on this
    # function as well, so we can reuse the images across runs.
    @staticmethod
    @st.cache(show_spinner=False)
    def load_image(url):
        # with urllib.request.urlopen(url) as response:
        #     image = np.asarray(bytearray(response.read()), dtype="uint8")
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # image = image[:, :, [2, 1, 0]] # BGR -> RGB
        image = cv2.imread(url)
        image = image[:, :, [2, 1, 0]]  # BGR -> RGB
        return image

    # Run the YOLO model to detect objects.

    # @st.cache(show_spinner=False)
    def rcnn_cropped(self, image, pred_class, bbox, output):
        printj.yellow(bbox)
        orig_copy = image.copy()
        printj.cyan(pred_class)
        if pred_class in ["outlet", "plumbing"]:
            crop_box = bbox.pad(pad_left=50, pad_right=50,
                                pad_top=50, pad_bottom=50)
            predict_dict = self.rcnn_OCR(
                orig_copy[crop_box.ymin: crop_box.ymax, crop_box.xmin: crop_box.xmax, :])
            score_list = predict_dict['score_list']
            bbox_list = predict_dict['bbox_list']
            pred_class_list = predict_dict['pred_class_list']
            for score, pred_class, bbox in zip(score_list, pred_class_list, bbox_list):
                if not self.confidence_threshold2:
                    self.confidence_threshold2 = 0.01
                if score > self.confidence_threshold2:
                    bbox = BBox(xmin=bbox.xmin+crop_box.xmin,
                                ymin=bbox.ymin+crop_box.ymin,
                                xmax=bbox.xmax+crop_box.xmin,
                                ymax=bbox.ymax+crop_box.ymin)
                    output = draw_bbox(img=output, bbox=bbox, color=255,
                                       show_label=True, text=RENAME2[pred_class], 
                                       thickness=2, text_size=1 )
                    # dist = np.linalg.norm(np.asarray(bbox.center) - np.asarray(crop_box.center))
                    output = cv2.line(output, bbox.center, crop_box.center, (150,150,150), 2) 
                    # show_image(output, 800)
        #             printj.red(bbox)
        # st.image(image=output.astype(np.uint8),)
        return output

    # @st.cache(show_spinner=False)
    def rcnn_OCR(self, image):
        weights_path = '/home/jitesh/3d/data/coco_data/ed/k/ElectricalDrawing/ElectricalDrawing/230_prediction_and_model_weight/weight_markchar/model_0013599.pth'
        inferer = D2Inferer(
            weights_path=weights_path,
            confidence_threshold=0.01,
            class_names=self.class_names2,
            model='Misc/cascade_mask_rcnn_R_50_FPN_3x',
            size_min=150, size_max=1000,
            detectron2_dir_path="/home/jitesh/prj/detectron2",
            gray_on=True,
        )
        predict_dict = inferer.predict(img=image)
        return predict_dict

    # @st.cache(show_spinner=False)
    def rcnn(self, image, confidence_threshold=0.01):
        weights_path = '/home/jitesh/3d/data/coco_data/ed/t2-5_coco-data/weights/faster_50_s2048_5/model_0001999.pth'
        weights_path = '/home/jitesh/3d/data/coco_data/ed/t2-12_coco-data/weights/faster_50_s1024_3/model_0002999.pth'
        weights_path = '/home/jitesh/3d/data/coco_data/ed/k/130_prediction_and_model_weight/weight/model_0001799.pth'
        size = 1024
        inferer = D2Inferer(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            # self.class_names=['t2','t3','t4','t8'],
            class_names=self.class_names,
            # model='COCO-Detection/faster_rcnn_R_50_FPN_1x',
            model='Misc/cascade_mask_rcnn_R_50_FPN_3x',
            size_min=2340, size_max=3500,
            detectron2_dir_path="/home/jitesh/prj/detectron2",
            gray_on=True,
            # crop_mode=3,
            # crop_mode3_sizes=[1024],
            # crop_mode3_overlaps=[100]
        )
        # inferer.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = overlap_threshold
        predict_dict = inferer.predict(img=image)
        # score_list = predict_dict['score_list']
        # bbox_list = predict_dict['bbox_list']
        # pred_class_list = predict_dict['pred_class_list']
        # pred_masks_list = predict_dict['pred_masks_list']
        # pred_keypoints_list = predict_dict['pred_keypoints_list']
        # vis_keypoints_list = predict_dict['vis_keypoints_list']
        # kpt_confidences_list = predict_dict['kpt_confidences_list']
        # boxes = pd.DataFrame()
        # boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
        return predict_dict
    # Run the YOLO model to detect objects.

    @staticmethod
    def yolo_v3(image, confidence_threshold, overlap_threshold):
        # Load the network. Because this is cached it will only happen once.
        @st.cache(allow_output_mutation=True)
        def load_network(config_path, weights_path):
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            output_layer_names = net.getLayerNames()
            output_layer_names = [output_layer_names[i[0] - 1]
                                  for i in net.getUnconnectedOutLayers()]
            return net, output_layer_names
        net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

        # Run the YOLO neural net.
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layer_names)

        # Supress detections in case of too low confidence or too much overlap.
        boxes, confidences, class_IDs = [], [], []
        H, W = image.shape[:2]
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)
                               ), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, overlap_threshold)

        # Map from YOLO labels to Udacity labels.
        UDACITY_LABELS = {
            0: 'pedestrian',
            1: 'biker',
            2: 'car',
            3: 'biker',
            5: 'truck',
            7: 'truck',
            9: 'trafficLight'
        }
        xmin, xmax, ymin, ymax, labels = [], [], [], [], []
        if len(indices) > 0:
            # loop over the indexes we are keeping
            for i in indices.flatten():
                label = UDACITY_LABELS.get(class_IDs[i], None)
                if label is None:
                    continue

                # extract the bounding box coordinates
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

                xmin.append(x)
                ymin.append(y)
                xmax.append(x+w)
                ymax.append(y+h)
                labels.append(label)

        boxes = pd.DataFrame(
            {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
        return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]


if __name__ == "__main__":
    STED(data_dir=IMG_DATA_PATH, image_names=IMG_NAMES,
         class_names=CLASS_NAMES, class_names2=CLASS_NAMES_OCR)
