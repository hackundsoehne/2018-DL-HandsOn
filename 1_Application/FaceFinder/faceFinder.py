import pyforms
from pyforms import BaseWidget
from pyforms.controls import ControlFile, ControlText, ControlButton, ControlList, ControlImage, ControlDir
import face_recognition
from face_recognition import face_recognition_cli
import PIL
import numpy as np
import glob
import os
import json
import cv2

class MainWindow(BaseWidget):
    def __init__(self):
        BaseWidget.__init__(self)
        self._known_face_box = ControlDir("Known Face Folder")
        self._search_folder_box = ControlDir("Image Folder")
        self._name_textbox = ControlText("Name")
        self._run_button = ControlButton("start facesearch")
        self._run_button.value = self._runButtonAction

        if os.path.exists('last_selection.json'):
            with open('last_selection.json', 'r') as f:
                d = json.load(f)
            self._known_face_box.value = d['known_face_folder']
            self._search_folder_box.value = d['search_folder']

    def _runButtonAction(self):
        known_face_folder = self._known_face_box.value
        search_folder = self._search_folder_box.value
        person_name = self._name_textbox
        with open('last_selection.json', 'w') as f:
            d = {
                'known_face_folder' : known_face_folder,
                'search_folder' : search_folder
            }
            json.dump(d, f)

        if os.path.isdir(search_folder) and os.path.isdir(known_face_folder):
            search_engine = FaceSearchEngine(known_face_folder)
            search_result = search_engine.run(search_folder)
            win = SearchResultWindow(search_result)
            win.show()


class SearchResultWindow(BaseWidget):
    def __init__(self, search_result):
        super(SearchResultWindow, self).__init__()
        self._file_list = ControlList("Search Result")
        self._file_list.horizontal_headers = ["path", 'known face']
        self._file_list.value = search_result
        self._file_list.select_entire_row = True
        self._file_list.readonly = True
        self._file_list.item_selection_changed_event = self._selection_changed_event
        self._name_text = ControlText('')

        self._image_viewer = ControlImage("Image Preview")

    def _selection_changed_event(self):
        # idx = self._file_list.selected_row_index
        path, name = self._file_list.get_currentrow_value()
        self._name_text.value = name
        self._image_viewer.value = cv2.imread(path, 1)



class FaceSearchEngine():
    def __init__(self, known_people_folder, cpus=1, tolerance=0.6):
        self._known_names, self._known_face_encodings = face_recognition_cli.scan_known_people(known_people_folder)
        self._cpus = cpus
        self._tolerance = tolerance

    def run(self, search_folder):
        image_files = sorted(glob.glob(os.path.join(search_folder, '*.jpg')))
        if self._cpus == 1:
            search_result_list = []
            for image_file in image_files:
                search_result = self._test_image(image_file)
                if search_result:
                    search_result_list.append(search_result)
            # search_result = [self._test_image(image_file) for image_file in image_files]
            # search_result = [t for t in search_result if t] # filter Nones
            #remove embedding from search result
            found_names = [name for embedding, name in search_result_list if name]
            search_result_list = [ (image_file, name) for image_file, name in zip(image_files, found_names)]
        return search_result_list



    def _test_image(self, image_file):
        '''
        give an image file path, the closest known_face_encoding and known_name is returned
        :param image_file:
        :return: closest known face encoding, closest known name
        '''
        unknown_image = face_recognition.load_image_file(image_file)

        # Scale down image if it's giant so things run a little faster
        unknown_image = self.maybeDownsize(unknown_image)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        for unknown_encoding in unknown_encodings:
            distances = face_recognition.face_distance(self._known_face_encodings, unknown_encoding)
            result = list(distances <= self._tolerance)
            if True in result: # we have a match
                closest_face_index = np.argmin(distances)
                return self._known_face_encodings[closest_face_index], self._known_names[closest_face_index]


    def run_multiprocess(self):
        pass

    def maybeDownsize(self, unknown_image):
        N = 512
        if max(unknown_image.shape) > N:
            pil_img = PIL.Image.fromarray(unknown_image)
            pil_img.thumbnail((N, N), PIL.Image.LANCZOS)
            unknown_image = np.array(pil_img)
        return unknown_image


# Execute the application
if __name__ == "__main__":     pyforms.start_app(MainWindow)