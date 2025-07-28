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
from chesscog.core import DEVICE, device
from chesscog.core.dataset import Datasets, build_transforms, name_to_piece
from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from PIL import Image
from recap import URI
from recap import CfgNode as CN
from roboflow_detection import detect

image_path = r"C:\chessPositions\chessBoard.png"
detect.get_chess_pieces(image_path)


class ChessRecognizer:
    """A class implementing the entire chess inference pipeline.

    Once you create an instance of this class, the CNNs are loaded into memory (possibly the GPU if available), so if you want to perform multiple inferences, they should all use one instance of this class for performance purposes.
    """

    _squares = list(chess.SQUARES)

    def __init__(self, classifiers_folder: Path = URI("models://")):
        """Constructor. Only loads corner detection config; skips all classifier loading."""
        self._corner_detection_cfg = CN.load_yaml_with_base(
            "config://corner_detection.yaml"
        )

        self._occupancy_cfg, self._occupancy_model = self._load_classifier(
            classifiers_folder / "occupancy_classifier"
        )
        self._occupancy_transforms = build_transforms(
            self._occupancy_cfg, mode=Datasets.TEST
        )
        self._pieces_cfg, self._pieces_model = self._load_classifier(
            classifiers_folder / "piece_classifier"
        )
        self._pieces_transforms = build_transforms(self._pieces_cfg, mode=Datasets.TEST)
        self._piece_classes = np.array(
            list(map(name_to_piece, self._pieces_cfg.DATASET.CLASSES))
        )

    @classmethod
    def _load_classifier(cls, path: Path):
        pt_files = list(path.glob("*.pt"))
        yaml_files = list(path.glob("*.yaml"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt model file found in {path}. Please ensure the required model file is present."
            )
        if not yaml_files:
            raise FileNotFoundError(
                f"No .yaml config file found in {path}. Please ensure the required config file is present."
            )
        model_file = pt_files[0]
        yaml_file = yaml_files[0]
        cfg = CN.load_yaml_with_base(yaml_file)
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        model = device(model)
        model.eval()
        return cfg, model

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

    def _classify_pieces(
        self,
        img: np.ndarray,
        turn: chess.Color,
        corners: np.ndarray,
        occupancy: np.ndarray,
    ) -> np.ndarray:
        occupied_squares = np.array(self._squares)[occupancy]
        warped = create_piece_dataset.warp_chessboard_image(img, corners)
        piece_imgs = map(
            functools.partial(create_piece_dataset.crop_square, warped, turn=turn),
            occupied_squares,
        )
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        piece_imgs = device(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=object)
        all_pieces[occupancy] = pieces
        return all_pieces

    def predict(
        self, img: np.ndarray, turn: chess.Color = chess.WHITE
    ) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

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

        # Obtain corners using the corner detection pipeline
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, img_scale = resize_image(self._corner_detection_cfg, img)
        corners = find_corners(self._corner_detection_cfg, img)

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
        camera_is_white_side: bool,
        visualise: bool = False,
    ) -> np.ndarray:
        # homography, from 3d image get a 2d warped image
        dst = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = os.path.join(tmpdirname, "raw_input.png")
            cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            try:
                detections = detect.get_chess_pieces(temp_path, show=False)
            except Exception as e:
                print(f"Roboflow detection failed: {e}")
                return np.full(len(self._squares), None, dtype=object)

        if detections is None or "predictions" not in detections:
            print("No predictions found in detection result.")
            return np.full(len(self._squares), None, dtype=object)

        # Warp image and zoom in
        warped = create_piece_dataset.warp_chessboard_image(img, corners)
        zoomed, zoomed_corners = self._zoom_in_on_center(warped, size=400)

        all_pieces = np.full(len(self._squares), None, dtype=object)
        vis_image = zoomed.copy()

        for det in detections["predictions"]:
            x_center = det["x"]
            y_center = det["y"]
            height = det["height"]
            width = det.get("width", height)

            adjusted_y = y_center + 0.3 * height
            original_point = np.array([[[x_center, adjusted_y]]], dtype=np.float32)
            warped_point = cv2.perspectiveTransform(original_point, M)[0][0]
            warped_x, warped_y = warped_point

            square_index = self._map_coords_to_square_index(
                warped_x, warped_y, zoomed_corners, camera_is_white_side
            )

            if 0 <= square_index < 64:
                piece = name_to_piece(det["class"])
                all_pieces[square_index] = piece

            # Draw bbox and point on zoomed image
            x1 = int(warped_x - width / 2)
            y1 = int(warped_y - height / 2)
            x2 = int(warped_x + width / 2)
            y2 = int(warped_y + height / 2)

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis_image, (int(warped_x), int(warped_y)), 5, (0, 0, 255), -1)

        # Draw 8x8 grid and square labels
        square_size = zoomed.shape[0] // 8

        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (200, 200, 200), 1)

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

        if visualise:
            # Show the raw input image before warping (for comparison)
            cv2.imshow("Raw Image Before Warp", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            # Show final detections on zoomed warped board
            cv2.imshow(
                "Detections on Zoomed Board", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return all_pieces

    def predict(
        self,
        img: np.ndarray,
        turn: chess.Color = chess.WHITE,
        camera_is_white_side: bool = True,
        visualise: bool = False,
    ) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference using only Roboflow for piece detection."""
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

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
    """Main method for running inference from the command line.

    Args:
        classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        setup (callable, optional): An optional setup function to be called after the CLI argument parser has been setup. Defaults to lambda:None.
    """

    parser = argparse.ArgumentParser(
        description="Run the chess recognition pipeline on an input image"
    )
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument(
        "--white",
        help="indicate that the image is from the white player's perspective (default)",
        action="store_true",
        dest="color",
    )
    parser.add_argument(
        "--black",
        help="indicate that the image is from the black player's perspective",
        action="store_false",
        dest="color",
    )
    parser.set_defaults(color=True)

    class Args:
        file = r"C:\chessPositions\0200.png"  # <-- your image path
        color = True  # True for white's perspective, False for black's

    args = Args()

    setup()

    img = cv2.imread(str(URI(image_path)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(
        img,
        chess.WHITE if args.color else chess.BLACK,
        camera_is_white_side=args.color,
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
