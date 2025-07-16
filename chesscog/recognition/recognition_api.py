import cv2
from pathlib import Path
from recap import URI
from .recognition import ChessRecognizer

def recognize_fen(image_path, classifiers_folder=Path("models")):
    """
    Runs chess recognition on an image and returns the FEN string.
    """
    img = cv2.imread(str(URI(image_path)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    recognizer = ChessRecognizer(classifiers_folder)
    board, _ = recognizer.predict(img)
    return board.board_fen()