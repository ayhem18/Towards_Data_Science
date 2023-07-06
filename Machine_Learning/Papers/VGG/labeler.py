"""
This scripts contains the functionality of the view of my small Labeler GUI application
"""
import shutil
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget, QLabel, QHBoxLayout,
)
from PyQt6.QtGui import QPixmap
from typing import Union
from pathlib import Path
import os

from functools import partial

HOME = os.getcwd()

SAVE_LABEL = 'save'
END_ITERATOR_INDEX = -1


def _process_path(path: Union[str, Path, None],
                  file_ok: bool = True,
                  dir_ok: bool = True,
                  create_ok: bool = False,
                  condition: callable = None) -> Union[str, Path, None]:
    if path is not None:
        # first make the save_path absolute
        path = path if os.path.isabs(path) else os.path.join(HOME, path)
        assert not \
            ((not file_ok and os.path.isfile(path)) or
             (not dir_ok and os.path.isdir(path))), \
            f'MAKE SURE NOT TO PASS A {"directory" if not dir_ok else "file"}'

        assert condition is None or condition(path), \
            "MAKE SURE THE passed path satisfies the condition passed with it"

        if create_ok and os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    return path


class LabelerWindow(QMainWindow):
    """
    The application's main window
    """
    # let's define some class specific constants
    WINDOW_SIZE = 800
    BUTTON_SHAPE = (50, 50)
    IMAGE_SIZE = (600, 600)

    @classmethod
    def __label_text(cls, index: int, image_path: Union[Path, str]) -> str:
        # first extract the basename
        image_name = os.path.basename(image_path)
        return f'image n: {index}\n {image_name}'

    def __init__(self, classes: list[str]):
        # those are the classes to choose from
        self.classes = classes
        self.image_index = 0

        # first call the super class' constructor
        super().__init__()
        # set a fixed size for the window
        self.setFixedSize(self.WINDOW_SIZE, self.WINDOW_SIZE)
        # let's set the layout for the application
        self.generalLayout = QVBoxLayout()

        # set the central widget
        parentWidget = QWidget(parent=self)
        parentWidget.setLayout(self.generalLayout)
        # don't forget to set it as the class' central widget
        self.setCentralWidget(parentWidget)

        # add the image placeholder
        self.__create_image_display()
        # add the buttons' placeholder
        self.__create_buttons()

    def __create_image_display(self):
        # the labels can be joined together in a single QVBoxLayout
        self.labelsLayout = QVBoxLayout()

        # the image's name label
        name_label = QLabel()
        name_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # the index's labels
        index_label = QLabel()
        index_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # the name label will be put on top of the index label
        self.labelsLayout.addWidget(name_label)
        self.labelsLayout.addWidget(index_label)

        # create the component to hold the image
        imagePart = QLabel()
        imagePart.setAlignment(Qt.AlignmentFlag.AlignLeft)
        imagePart.setFixedSize(*self.IMAGE_SIZE)

        # create the layout to hold both the label and the image
        self.imageLayout = QHBoxLayout()
        self.imageLayout.addLayout(self.labelsLayout)
        self.imageLayout.addWidget(imagePart)

        # add the imageLayout to the general Layout
        self.generalLayout.addLayout(self.imageLayout)

    def __create_buttons(self):
        # the main idea here is to create len(classes) + 1 buttons
        # the first button is a 'save' button
        # the rest are buttons with the class names
        self.buttonsLayout = QHBoxLayout()
        # define the save button separately
        self.buttonMap = {SAVE_LABEL: QPushButton(text=SAVE_LABEL)}
        self.buttonMap[SAVE_LABEL].setFixedSize(*self.BUTTON_SHAPE)
        # don't forget to add it to the layout
        self.buttonsLayout.addWidget(self.buttonMap[SAVE_LABEL])

        for cls in self.classes:
            self.buttonMap[cls] = QPushButton(text=cls.strip().capitalize())
            self.buttonMap[cls].setFixedSize(*self.BUTTON_SHAPE)
            # add the button to the layout
            self.buttonsLayout.addWidget(self.buttonMap[cls])

        # add the buttons layout to the general layout
        self.generalLayout.addLayout(self.buttonsLayout)

    def display_image(self, index: int, image_path: Union[Path, str]):
        if index == END_ITERATOR_INDEX:
            self.close()
            sys.exit()
        # set the text for
        self.imageLayout.itemAt(0).layout().itemAt(0).widget().setText(f"image: {os.path.basename(image_path)}")
        self.imageLayout.itemAt(0).layout().itemAt(1).widget().setText(f"image's index: {index}")

        # the first step is to set the text of the label
        # self.imageLayout.itemAt(0).widget() \
        #     .setText(self.__label_text(index, image_path))

        # the second is to display the image
        pixels = QPixmap(image_path)
        self.imageLayout.itemAt(1).widget().setPixmap(pixels)


class LabelerModel:
    """
    This is the Model class for the Labeler GUI application
    """

    def __init__(self,
                 classes: list[str],
                 from_directory: Union[str, Path],
                 to_directory: Union[str, Path],
                 copy: bool = True,
                 batch_size: Union[int, float] = 0.1):
        self.classes = classes
        # first process the given paths
        self.from_dir = _process_path(from_directory, file_ok=False)
        self.to_dir = _process_path(to_directory, file_ok=False, create_ok=True)

        # initialize the folder corresponding to each class
        for c in self.classes:
            folder_path = os.path.join(self.to_dir, c)
            os.makedirs(folder_path, exist_ok=True)

        # boolean flag to determine whether to copy the files or move them
        self.copy = copy
        # the number of files to be copied / cut at once
        self.batch_size = len(os.listdir(self.from_dir)) * batch_size if isinstance(batch_size, float) else batch_size
        # a generator to return
        self.generator = self.__path_generator()
        self.current_index = None
        self.current_path = None
        self.information = {}

    def __path_generator(self):
        for index, file_name in enumerate(os.listdir(self.from_dir)):
            path_name = os.path.join(self.from_dir, file_name)
            self.current_index = index
            self.current_path = path_name
            yield index, path_name

    def get(self) -> tuple[int, Union[Path, str, None]]:
        try:
            return next(self.generator)
        except StopIteration:
            return END_ITERATOR_INDEX, None

    def save_files(self):
        func = (lambda path, new_path: shutil.move(path, new_path)) if not self.copy else \
            (lambda path, new_path: shutil.copy(path, new_path))

        for _, (path, cls) in self.information.items():
            new_path = os.path.join(self.to_dir, cls, os.path.basename(path))
            func(path, new_path)

        # don't forget to clear the dictionary
        self.information.clear()

    def process_class(self, cls: str):
        assert cls in self.classes, "THE PASSED CLASS IS NOT IN THE PREDEFINED LIST OF CLASSES"
        # cls corresponds to the image with index = current_index
        self.information[self.current_index] = (self.current_path, cls)

        if len(self.information) >= self.batch_size:
            self.save_files()


## time to create the Controller

class LabelerController:
    def __init__(self, view: LabelerWindow, model: LabelerModel):
        self.view = view
        self.model = model
        # the first thing to pass the index and the path name to the view from the model
        index, path = self.model.get()
        self.view.display_image(index, path)
        self._connectSignalsAndSlots()

    # let's define the functions that need to be done here
    def __connect_save_button(self):
        # hitting the save button means, copying / moving
        # all the files that are currently
        # saved in the dictionary
        self.model.save_files()

    def __connect_cls_button(self, cls):
        # first set the class to the model
        self.model.process_class(cls)
        # get the next image
        index, path = self.model.get()
        self.view.display_image(index, path)

    def _connectSignalsAndSlots(self):
        # connect the save button first
        self.view.buttonMap[SAVE_LABEL].clicked.connect(self.__connect_save_button)
        for key, button in self.view.buttonMap.items():
            if key != SAVE_LABEL:
                # since __connect_cls_button accepts an argument
                # we need to create a partial function for each class
                partial_func = partial(self.__connect_cls_button, key)
                self.view.buttonMap[key].clicked.connect(partial_func)


def main():
    classes = ['dog', 'cat']
    from_dir = os.path.join(os.getcwd(), 'test_images')
    to_dir = os.path.join(os.getcwd(), 'test_labels')
    labelerApp = QApplication([])
    view = LabelerWindow(classes=classes)
    # let's initialize the controller
    model = LabelerModel(classes=classes,
                         from_directory=from_dir,
                         to_directory=to_dir,
                         copy=False,
                         )
    controller = LabelerController(view=view, model=model)
    view.show()
    sys.exit(labelerApp.exec())


if __name__ == '__main__':
    main()
