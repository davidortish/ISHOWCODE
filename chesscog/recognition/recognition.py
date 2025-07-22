"""Module that brings together the whole recognition pipeline into a single class so it can be conveniently executed.

This module simultaneously acts as a script to perform a single inference:

.. code-block:: console

    $ python -m chesscog.recognition.recognition --help
    usage: recognition.py [-h] [--white] [--black] file

    Run the chess recognition pipeline on an input image

    positional arguments:
      file        path to the input image

    optional arguments:
      -h, --help  show this help message and exit
      --white     indicate that the image is from the white player's
                  perspective (default)
      --black     indicate that the image is from the black player's
                  perspective
"""

import argparse
import functools
import os
import tempfile
import typing
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from chess import Status
from PIL import Image
from recap import URI
from recap import CfgNode as CN

from chesscog.core import DEVICE, device
from chesscog.core.dataset import Datasets, build_transforms, name_to_piece
from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from roboflow_detection import detect


class ChessRecognizer:
    """A class implementing the entire chess inference pipeline.

    Once you create an instance of this class, the CNNs are loaded into memory (possibly the GPU if available), so if you want to perform multiple inferences, they should all use one instance of this class for performance purposes.
    """

    _squares = list(chess.SQUARES)

    def __init__(self, classifiers_folder: Path = URI("models://")):
        """Constructor.

        Args:
            classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        """
        self._corner_detection_cfg = CN.load_yaml_with_base(
            "config://corner_detection.yaml"
        )

        # Load occupancy classifier (still used)
        self._occupancy_cfg, self._occupancy_model = self._load_classifier(
            classifiers_folder / "occupancy_classifier"
        )
        self._occupancy_transforms = build_transforms(
            self._occupancy_cfg, mode=Datasets.TEST
        )

        # REMOVE PIECE CLASSIFIER: we use Roboflow instead
        # self._pieces_cfg, self._pieces_model = self._load_classifier(
        #     classifiers_folder / "piece_classifier")
        # self._pieces_transforms = build_transforms(
        #     self._pieces_cfg, mode=Datasets.TEST)
        # self._piece_classes = np.array(list(map(name_to_piece,
        #                                         self._pieces_cfg.DATASET.CLASSES)))

    @staticmethod
    def order_corners(corners: np.ndarray) -> np.ndarray:
        """
        Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left.
        This is necessary for perspective warping to behave correctly.
        """
        corners = corners.reshape(4, 2)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        top_left = corners[np.argmin(s)]
        bottom_right = corners[np.argmax(s)]
        top_right = corners[np.argmin(diff)]
        bottom_left = corners[np.argmax(diff)]

        return np.array(
            [top_left, top_right, bottom_right, bottom_left], dtype="float32"
        )

    @classmethod
    def _load_classifier(cls, path: Path):
        model_file = next(iter(path.glob("*.pt")))
        yaml_file = next(iter(path.glob("*.yaml")))
        cfg = CN.load_yaml_with_base(yaml_file)
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        model = device(model)
        model.eval()
        return cfg, model

    def _zoom_center(self, image: np.ndarray, zoom_factor: float = 2.0) -> np.ndarray:
        """
        Zoom into the center of the image by the given zoom factor.

        Args:
            image: RGB image as a NumPy array.
            zoom_factor: Factor to zoom (e.g., 2.0 zooms into the center 1/2 region).

        Returns:
            Zoomed-in image with same dimensions.
        """
        h, w = image.shape[:2]
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        cropped = image[top : top + new_h, left : left + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return zoomed

    def _zoom_in_on_center(self, img: np.ndarray, size: int = 600):
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2
        half = size // 2

        # Crop image
        cropped = img[cy - half : cy + half, cx - half : cx + half]

        # The original warped board corners are at (0,0), (w,0), (w,h), (0,h)
        # After cropping, the new corners shift by (cx - half, cy - half)
        new_corners = np.array(
            [[0, 0], [size, 0], [size, size], [0, size]], dtype="float32"
        )

        return cropped, new_corners

    def _classify_occupancy(
        self, img: np.ndarray, turn: chess.Color, corners: np.ndarray
    ) -> np.ndarray:
        warped = create_occupancy_dataset.warp_chessboard_image(img, corners)
        square_imgs = map(
            functools.partial(create_occupancy_dataset.crop_square, warped, turn=turn),
            self._squares,
        )
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1
        ) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy

    def _build_board_from_roboflow(
        self, image_path: str, corners: np.ndarray, turn: chess.Color
    ) -> chess.Board:
        """
        Build a chess.Board from Roboflow detection results using piece locations and types.

        Args:
            image_path (str): Path to the input image.
            corners (np.ndarray): 4 corner points of the board.
            turn (chess.Color): Current player's turn.

        Returns:
            chess.Board: A board populated with the detected pieces.
        """
        result = detect.get_chess_pieces(image_path, show=False)
        board = chess.Board()
        board.clear()
        board.turn = turn

        label_to_piece = {
            "white-pawn": chess.Piece(chess.PAWN, chess.WHITE),
            "white-rook": chess.Piece(chess.ROOK, chess.WHITE),
            "white-knight": chess.Piece(chess.KNIGHT, chess.WHITE),
            "white-bishop": chess.Piece(chess.BISHOP, chess.WHITE),
            "white-queen": chess.Piece(chess.QUEEN, chess.WHITE),
            "white-king": chess.Piece(chess.KING, chess.WHITE),
            "black-pawn": chess.Piece(chess.PAWN, chess.BLACK),
            "black-rook": chess.Piece(chess.ROOK, chess.BLACK),
            "black-knight": chess.Piece(chess.KNIGHT, chess.BLACK),
            "black-bishop": chess.Piece(chess.BISHOP, chess.BLACK),
            "black-queen": chess.Piece(chess.QUEEN, chess.BLACK),
            "black-king": chess.Piece(chess.KING, chess.BLACK),
        }

        for det in result["predictions"]:
            class_name = det["class"]
            x_center = det["x"]
            y_center = det["y"]

            square = self._map_coords_to_square_index(x_center, y_center, corners, turn)
            if square == -1:
                continue  # skip invalid locations

            piece = label_to_piece.get(class_name)
            if piece:
                board.set_piece_at(square, piece)

        return board

    def _map_coords_to_square_index(
        self, x: float, y: float, corners: np.ndarray, camera_is_white_side: bool
    ) -> int:
        # Define the standard 800x800 board layout for warping
        board_pts = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype="float32")

        # Compute perspective transform from image corners to board coordinates
        matrix = cv2.getPerspectiveTransform(corners.astype("float32"), board_pts)

        # Apply the perspective transform to the point
        warped_point = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype="float32"), matrix
        )[0][0]

        # Determine which square the point maps to
        square_size = 100  # since 800 / 8 = 100
        col = int(warped_point[0] // square_size)
        row = int(warped_point[1] // square_size)

        if not (0 <= row < 8 and 0 <= col < 8):
            return -1  # point lies outside the board

        # Convert to square index (0–63) based on camera's perspective
        if camera_is_white_side:
            square_index = row * 8 + col
        else:
            square_index = (7 - row) * 8 + (7 - col)

        return square_index

    def _flip_board_horizontally(self, board: chess.Board) -> chess.Board:
        flipped_board = chess.Board.empty()
        flipped_board.turn = board.turn
        flipped_board.clear_board()

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                # flip horizontally: file -> 7 - file, same rank
                flipped_square = chess.square(7 - file, rank)
                flipped_board.set_piece_at(flipped_square, piece)
        return flipped_board

    def _classify_pieces(
        self,
        img: np.ndarray,
        turn: chess.Color,
        corners: np.ndarray,
        occupancy: np.ndarray,
        camera_is_white_side: bool,
    ) -> np.ndarray:
        # Warp the original image to get top-down board view
        warped = create_piece_dataset.warp_chessboard_image(img, corners)

        # Zoom in on the center of the warped image (adjust size as needed)
        zoomed, zoomed_corners = self._zoom_in_on_center(warped, size=400)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = os.path.join(tmpdirname, "zoomed_warped.png")
            cv2.imwrite(temp_path, cv2.cvtColor(zoomed, cv2.COLOR_RGB2BGR))
            try:
                detections = detect.get_chess_pieces(temp_path, show=False)
            except Exception as e:
                print(f"Roboflow detection failed: {e}")
                return np.full(len(self._squares), None, dtype=object)

        if detections is None or "predictions" not in detections:
            print("No predictions found in detection result.")
            return np.full(len(self._squares), None, dtype=object)

        all_pieces = np.full(len(self._squares), None, dtype=object)

        # Visualization image in RGB for drawing
        vis_image = zoomed.copy()

        for det in detections["predictions"]:
            x_center = det["x"]
            y_center = det["y"]
            height = det["height"]
            width = det.get("width", height)  # fallback width if not given

            # Adjust y coordinate to 0.8 of height from top (same as before)
            adjusted_y = y_center + 0.3 * height

            # Map using perspective parameter with zoomed corners now
            square_index = self._map_coords_to_square_index(
                x_center, adjusted_y, zoomed_corners, camera_is_white_side
            )

            print(
                f"Detected {det['class']} at ({x_center:.2f}, {adjusted_y:.2f}) mapped to square index {square_index}"
            )
            if 0 <= square_index < 64:
                piece = name_to_piece(det["class"])
                all_pieces[square_index] = piece

            # Draw bbox and adjusted point for debugging
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis_image, (int(x_center), int(adjusted_y)), 5, (0, 0, 255), -1)

        # Draw 8x8 grid and square labels on the zoomed image
        square_size = zoomed.shape[0] // 8  # should be 50 if size=400

        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (200, 200, 200), 1)

                # Compute square index considering camera perspective
                square_idx = row * 8 + col
                if not camera_is_white_side:
                    square_idx = (7 - row) * 8 + (7 - col)
                square_name = chess.square_name(square_idx)
                cv2.putText(
                    vis_image,
                    square_name,
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return all_pieces

    def predict(
        self,
        img: np.ndarray,
        turn: chess.Color = chess.WHITE,
        camera_is_white_side: bool = True,
    ) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)

            # Find and order the board corners
            raw_corners = find_corners(self._corner_detection_cfg, img)
            corners = self.order_corners(raw_corners)

            # Detect which squares are occupied
            occupancy = self._classify_occupancy(img, turn, corners)

            # Classify pieces using Roboflow
            pieces = self._classify_pieces(
                img, turn, corners, occupancy, camera_is_white_side
            )

            # Build the board
            board = chess.Board()
            board.clear_board()
            for square_index, piece in enumerate(pieces):
                if piece:
                    board.set_piece_at(square_index, piece)

            # Return scaled-back corners
            corners = corners / img_scale
            return board, corners


class TimedChessRecognizer(ChessRecognizer):
    """A subclass of :class:`ChessRecognizer` that additionally records the time taken for each step of the pipeline during inference."""

    def predict(
        self,
        img: np.ndarray,
        turn: chess.Color = chess.WHITE,
        camera_is_white_side: bool = True,
    ) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        from timeit import default_timer as timer

        with torch.no_grad():
            t1 = timer()
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            t2 = timer()
            occupancy = self._classify_occupancy(img, turn, corners)
            t3 = timer()
            pieces = self._classify_pieces(
                img, turn, corners, occupancy, camera_is_white_side
            )
            t4 = timer()

            board = chess.Board()
            board.clear()
            board.turn = turn
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            t5 = timer()

            times = {
                "corner_detection": t2 - t1,
                "occupancy_classification": t3 - t2,
                "piece_classification": t4 - t3,
                "prepare_results": t5 - t4,
            }

            return board, corners, times


def main(classifiers_folder: Path = URI("models://"), setup: callable = lambda: None):
    # === Manually set your input values here ===
    image_path = r"C:\chessPositions\0254.png"  # <<-- CHANGE THIS
    camera_white = False  # Set to False if image is from black's perspective
    turn_white = False  # Set to False if it's black's turn

    setup()

    img = cv2.imread(str(URI(image_path)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(
        img,
        chess.WHITE if turn_white else chess.BLACK,
        camera_is_white_side=camera_white,
    )

    flipped_board = recognizer._flip_board_horizontally(board)
    print(flipped_board)
    print()
    print(
        f"You can view this position at https://lichess.org/editor/{flipped_board.board_fen()}"
    )


if __name__ == "__main__":
    from chesscog.occupancy_classifier.download_model import (
        ensure_model as ensure_occupancy_classifier,
    )
    from chesscog.piece_classifier.download_model import (
        ensure_model as ensure_piece_classifier,
    )

    main(
        setup=lambda: [
            ensure_model(show_size=True)
            for ensure_model in (ensure_occupancy_classifier, ensure_piece_classifier)
        ]
    )
